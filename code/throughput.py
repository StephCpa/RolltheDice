"""P6 WP2 (C01): teacherless 集合代价 vs 朴素 m² 前向 的吞吐微基准。

问题：为 G 个组、每组 m 个 seed × m 个候选解，计算全部 m² 配对代价：
- teacherless 路线：每 seed 一次前向（共 G×m 条序列），m² 个代价 = logits gather；
- 朴素自回归路线：每个 (seed, 解) 一条前向（共 G×m² 条序列），答案段 CE 求和。
两者都按 64 序列/批分块、pad 到 128，硬件与 dtype 相同。输出加速比。
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

MAX_LEN = 128


def synth_circle_like(tok, n, eos, L_ans=28, L_prefix=10, seed=0):
    """合成 circle 形状的序列：prefix 用随机 token、答案段随机 vocab token。"""
    rng = np.random.default_rng(seed)
    V = len(tok)
    seqs = []
    for _ in range(n):
        p = rng.integers(0, V, L_prefix).tolist()
        a = rng.integers(0, V, L_ans).tolist()
        seqs.append((p, a))
    return seqs


def forward_batch(model, seq_pairs, eos, device, chunk=64, mtp=True):
    """mtp=True: 答案位填 eos（teacherless 输入）；False: 填真实答案（朴素 AR 输入）。
    返回 logprobs [B, MAX_LEN, V]（fp32, no_grad）。"""
    outs = []
    with torch.no_grad():
        for i in range(0, len(seq_pairs), chunk):
            batch = seq_pairs[i:i + chunk]
            inp = torch.full((len(batch), MAX_LEN), eos, dtype=torch.long)
            for b, (p, a) in enumerate(batch):
                ids = p + ([eos] * (len(a) - 1) + a[-1:] if mtp else a)
                L = min(len(ids), MAX_LEN)
                inp[b, :L] = torch.tensor(ids[:L])
            inp = inp.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(device == "cuda")):
                logits = model(inp).logits
            outs.append(F.log_softmax(logits.float(), dim=-1).cpu())
    return torch.cat(outs)


def cost_teacherless(model, groups, eos, device):
    """groups: list of m 个 (p, a)。G×m 次前向 + gather 得 G×m×m 代价。"""
    flat = [ex for grp in groups for ex in grp]
    lp = forward_batch(model, flat, eos, device, mtp=True)
    G, m = len(groups), len(groups[0])
    C = np.zeros((G, m, m))
    for g in range(G):
        L = len(groups[g][0][1])
        for i in range(m):
            b = g * m + i
            P = len(groups[g][i][0])
            sl = lp[b, P - 1:P - 1 + L, :]
            for j in range(m):
                tgt = torch.tensor(groups[g][j][1])
                C[g, i, j] = -sl[torch.arange(L), tgt].sum().item()
    return C


def cost_naive(model, groups, eos, device):
    """每个 (seed_i, 解_j) 一条 NTP 输入前向：G×m² 条序列。"""
    pairs = []
    meta = []
    for g, grp in enumerate(groups):
        for i in range(len(grp)):
            for j in range(len(grp)):
                pairs.append((grp[i][0], grp[j][1]))
                meta.append((g, i, j))
    lp = forward_batch(model, pairs, eos, device, mtp=False)
    G, m = len(groups), len(groups[0])
    C = np.zeros((G, m, m))
    for (g, i, j), (p, a), b in zip(meta, pairs, range(len(pairs))):
        P = len(p)
        sl = lp[b, P - 1:P - 1 + len(a), :]
        tgt = torch.tensor(a)
        C[g, i, j] = -sl[torch.arange(len(a)), tgt].sum().item()
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--vocab_extra", type=int, default=40, help="模拟 add_tokens 后词表增量")
    ap.add_argument("--G", type=int, default=16)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    tok.add_tokens([f"<x{i}>" for i in range(args.vocab_extra)])
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.resize_token_embeddings(len(tok))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    eos = tok.eos_token_id

    groups = []
    for g in range(args.G):
        seqs = synth_circle_like(tok, args.m, eos, seed=g)
        groups.append(seqs)

    # 代价分歧记录（非等价验证）：teacherless 代价（dummy 槽并行预测）与
    # 朴素 AR 代价（真实前缀 teacher forcing）本就是不同条件似然——前者是
    # 训练代理目标（方案 §五.2）。此处量化两者分歧，作为"代理目标≠完整序列概率"的实测。
    C1 = cost_teacherless(model, groups, eos, device)
    C2 = cost_naive(model, groups, eos, device)
    diff = float(np.abs(C1 - C2).max())
    print("max |C_teacherless - C_naive| =", diff)

    # 计时
    def bench(fn):
        ts = []
        for r in range(args.reps):
            t0 = time.time()
            fn()
            ts.append(time.time() - t0)
        return float(np.median(ts)), float(np.mean(ts))

    t_tl_med, t_tl_mean = bench(lambda: cost_teacherless(model, groups, eos, device))
    t_nv_med, t_nv_mean = bench(lambda: cost_naive(model, groups, eos, device))
    res = {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "G": args.G, "m": args.m, "reps": args.reps,
        "cost_divergence_max_abs": diff,
        "note2": "teacherless 代价与 AR 代价是不同条件似然（代理目标），差异为预期",
        "teacherless_sec_median": t_tl_med,
        "naive_sec_median": t_nv_med,
        "speedup_median": t_nv_med / t_tl_med,
        "teacherless_sec_mean": t_tl_mean,
        "naive_sec_mean": t_nv_mean,
        "speedup_mean": t_nv_mean / t_tl_mean,
        "note": "G×m 次 teacherless 前向 + gather vs G×m² 条朴素前向；批大小 64、seq 128",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
