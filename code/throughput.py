"""P6 C01 微基准（按评审第 4–5 轮拆分）。

C01a（同一目标的计算复用，C01 主张本体）：同一 teacherless 代价矩阵的两种实现——
  复用：每个 seed 一条 teacherless 序列（G·m 条），m² 个代价 = 同一分布上的 gather；
  重算：每个 (seed_i, 解_j) 一条 teacherless 序列（G·m² 条，输入 p_i + dummy + a_j[-1]），只取 a_j 的代价。
  先核对两者矩阵一致（eval 模式；fp16 下只差内核舍入），再计时。
C01b（跨目标路线比，不是同目标加速）：teacherless 代价 vs AR（teacher-forcing）代价 G·m² 条序列；
  报告决策一致率 P(π_TL = π_AR)、AR 意义下的遗憾，以及只影响双射选择的交互项（双中心化）差异——
  原始逐元最大差不能说明决策不同（C+a_i+b_j 不改变最优配对）。
口径：序列数、实际 token 数与 padding 后 token 数、前向 FLOPs 近似（2·非嵌入参数·token），
GPU 上批量（--bs）、gather 在 GPU 上完成、计时前后 cuda.synchronize。
注意：这是代价矩阵计算的基准；整步训练的耗时还含反向与 NTP 前向，须另以 throughput.json 报告。

用法：python code/throughput.py --model runs/x/final_model --data_dir data/circle.10.9.10.10000 \
        --m 4,8,16 --G 16 --out c01.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import assign, build_index_groups, encode_raw  # noqa: E402


def make_groups(args, tok, m, G):
    if args.data_dir:
        raw = json.load(open(os.path.join(args.data_dir, "train.json"), encoding="utf-8"))
        exs = encode_raw(raw[: max(G * m * 4, 64)], tok)
        idx = build_index_groups(len(exs), m, np.random.default_rng(0))[:G]
        return [[exs[j] for j in mem] for mem in idx]
    rng = np.random.default_rng(0)
    V = len(tok)
    return [[(rng.integers(0, V, 8).tolist(), rng.integers(0, V, 27).tolist())
             for _ in range(m)] for _ in range(G)]


def run_batches(model, seqs, gather_fn, device, bs, fp16):
    """seqs: [(ids, P, meta)]；gather_fn(logp_rows[b, L, V], metas) → list。返回 (结果, 实际 token, padded token)。"""
    out, real, padded = [], 0, 0
    eos = model.config.eos_token_id
    with torch.no_grad():
        for c0 in range(0, len(seqs), bs):
            chunk = seqs[c0:c0 + bs]
            T = max(len(s[0]) for s in chunk)
            inp = torch.full((len(chunk), T), eos, dtype=torch.long)
            for b, (ids, _, _) in enumerate(chunk):
                inp[b, :len(ids)] = torch.tensor(ids)
            real += sum(len(s[0]) for s in chunk)
            padded += len(chunk) * T
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(fp16 and device == "cuda")):
                logits = model(inp.to(device)).logits
            out.extend(gather_fn(logits, chunk))
    return out, real, padded


def _rows(logits, chunk, L, device):
    P = torch.tensor([s[1] for s in chunk], device=device)
    pos = P.unsqueeze(1) - 1 + torch.arange(L, device=device)            # [b, L]
    b = torch.arange(len(chunk), device=device).unsqueeze(1).expand(-1, L)
    return F.log_softmax(logits[b, pos].float(), dim=-1)                   # [b, L, V]


def costs_reuse(model, groups, end_id, device, bs, fp16):
    G, m = len(groups), len(groups[0])
    L = len(groups[0][0][1])
    eos = model.config.eos_token_id
    tgt = torch.tensor([[a for _, a in grp] for grp in groups], device=device)   # [G, m, L]
    seqs = [(p + [eos] * (L - 1) + [end_id], len(p), (g, i))
            for g, grp in enumerate(groups) for i, (p, _) in enumerate(grp)]

    def gather(logits, chunk):
        lp = _rows(logits, chunk, L, device)
        gs = torch.tensor([s[2][0] for s in chunk], device=device)
        idx = tgt[gs].transpose(1, 2)                                      # [b, L, m]
        return list((-lp.gather(2, idx).sum(1)).cpu().numpy())             # [b, m]
    res, real, padded = run_batches(model, seqs, gather, device, bs, fp16)
    return np.array(res).reshape(G, m, m), len(seqs), real, padded


def costs_recompute(model, groups, end_id, device, bs, fp16):
    G, m = len(groups), len(groups[0])
    L = len(groups[0][0][1])
    eos = model.config.eos_token_id
    seqs = [(p + [eos] * (L - 1) + grp[j][1][-1:], len(p), grp[j][1])
            for grp in groups for (p, _) in grp for j in range(m)]

    def gather(logits, chunk):
        lp = _rows(logits, chunk, L, device)
        idx = torch.tensor([s[2] for s in chunk], device=device).unsqueeze(2)   # [b, L, 1]
        return list((-lp.gather(2, idx).sum((1, 2))).cpu().numpy())
    res, real, padded = run_batches(model, seqs, gather, device, bs, fp16)
    return np.array(res).reshape(G, m, m), len(seqs), real, padded


def costs_ar(model, groups, device, bs, fp16):
    G, m = len(groups), len(groups[0])
    L = len(groups[0][0][1])
    seqs = [(p + grp[j][1], len(p), grp[j][1])
            for grp in groups for (p, _) in grp for j in range(m)]

    def gather(logits, chunk):
        lp = _rows(logits, chunk, L, device)
        idx = torch.tensor([s[2] for s in chunk], device=device).unsqueeze(2)
        return list((-lp.gather(2, idx).sum((1, 2))).cpu().numpy())
    res, real, padded = run_batches(model, seqs, gather, device, bs, fp16)
    return np.array(res).reshape(G, m, m), len(seqs), real, padded


def timed(fn, reps, device):
    ts = []
    for _ in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))


def double_center(M):
    return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", help="gpt2 或训练好的 final_model / ckpts 目录")
    ap.add_argument("--data_dir", default=None, help="真实数据（缺省用合成序列）")
    ap.add_argument("--vocab_extra", type=int, default=16, help="--model=gpt2 且无 data_dir 时模拟 add_tokens")
    ap.add_argument("--m", default="4,8,16")
    ap.add_argument("--G", type=int, default=16)
    ap.add_argument("--bs", type=int, default=256, help="每次前向的序列数")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--fp32", action="store_true", help="关闭 fp16 autocast（一致性核对可到逐位）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    a_id = tok.convert_tokens_to_ids("</a>")
    if a_id is None or a_id == tok.unk_token_id:   # 原始 gpt2：补上任务 token（训练好的检查点已含）
        new = json.load(open(os.path.join(args.data_dir, "vocab.json"), encoding="utf-8")) \
            if args.data_dir else [f"<x{i}>" for i in range(args.vocab_extra)] + ["</a>"]
        tok.add_tokens(new)
        model.resize_token_embeddings(len(tok))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    end_id = tok.convert_tokens_to_ids("</a>")
    fp16 = not args.fp32
    n_nonemb = sum(p.numel() for n, p in model.named_parameters() if "wte" not in n and "wpe" not in n)

    results = []
    for m in [int(x) for x in args.m.split(",")]:
        groups = make_groups(args, tok, m, args.G)
        (C_re, n_re, r_re, p_re), t_re = timed(
            lambda: costs_reuse(model, groups, end_id, device, args.bs, fp16), args.reps, device)
        (C_rc, n_rc, r_rc, p_rc), t_rc = timed(
            lambda: costs_recompute(model, groups, end_id, device, args.bs, fp16), args.reps, device)
        (C_ar, n_ar, r_ar, p_ar), t_ar = timed(
            lambda: costs_ar(model, groups, device, args.bs, fp16), args.reps, device)
        agree, regret = [], []
        for g in range(len(groups)):
            p_tl, p_a = assign(C_re[g]), assign(C_ar[g])
            agree.append(p_tl == p_a)
            regret.append(C_ar[g][range(m), p_tl].sum() - C_ar[g][range(m), p_a].sum())
        res = {
            "m": m, "G": len(groups), "bs": args.bs, "fp16": fp16, "device":
            torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "C01a_same_objective": {
                "n_seq_reuse": n_re, "n_seq_recompute": n_rc,
                "tokens_reuse": r_re, "tokens_recompute": r_rc,
                "padded_tokens_reuse": p_re, "padded_tokens_recompute": p_rc,
                "fwd_gflops_reuse": 2 * n_nonemb * r_re / 1e9,
                "fwd_gflops_recompute": 2 * n_nonemb * r_rc / 1e9,
                "sec_reuse_median": t_re, "sec_recompute_median": t_rc,
                "speedup": t_rc / t_re,
                "consistency_max_abs_nats": float(np.max(np.abs(C_re - C_rc))),
                "consistency_same_argmin_frac": float(np.mean(
                    [assign(C_re[g]) == assign(C_rc[g]) for g in range(len(groups))]))},
            "C01b_cross_objective": {
                "n_seq_ar": n_ar, "tokens_ar": r_ar, "sec_ar_median": t_ar,
                "time_ratio_ar_over_tl_reuse": t_ar / t_re,
                "P_piTL_eq_piAR": float(np.mean(agree)),
                "chance": 1.0 / float(np.prod(range(1, m + 1))),
                "ar_regret_nats_mean": float(np.mean(regret)),
                "ar_regret_nats_median": float(np.median(regret)),
                "max_abs_raw_diff": float(np.max(np.abs(C_re - C_ar))),
                "max_abs_interaction_diff": float(max(np.max(np.abs(double_center(C_re[g]) -
                                                                    double_center(C_ar[g])))
                                                      for g in range(len(groups))))}}
        results.append(res)
        print(json.dumps(res, indent=1), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "data_dir": args.data_dir,
                   "note": "代价矩阵计算基准；C01a 为同目标复用加速，C01b 为跨目标路线比与决策一致性",
                   "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
