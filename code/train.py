"""P6 WP1/WP2: 干净重实现的训练器（五条臂）。

语义来源（逐条对照 chenwu98/algorithmic-creativity @7176dc7）：
- 序列构造 / 损失掩码: simpletransformers/seq2seq/seq2seq_utils.py::preprocess_data_bart
  （input_ids=target_ids 起始于 prefix；labels 前缀与 pad 置 -100；
  MTP 项输入在答案位为 eos pad、标签为真值 → teacherless 并行预测，因果注意下无泄漏）。
- 混合比: *_hybrid.ipynb（每例 1×NTP + 3×MTP）。
- 优化: train loop（AdamW, clip 1.0, fp16 AMP, constant_schedule_with_warmup(2000)）；
  train.sh（gpt2, seq 128, bs 64, lr 1e-4, wd 0.0）。
- 评价频率: 每 5000 步 predict。

臂（--arm）:
  ntp_noseed   无 seed NTP（no-hash 数据）            64 例/步
  ntp_fixed    固定 seed 配对 NTP（原论文设定）        64 例/步
  ntp_randperm G1 对照: 固定 seed 集＋每轮随机双射     64 例/步
  tl_hybrid    原 teacherless 混合（1 NTP+3 MTP）     16 例/步（忠实原文展开）
  setc_m4      WP2 集合耦合（teacherless 代价+Hungarian，
               梯度只走被配对条目＋NTP 混合×3）        64 例/步
主要对照 fixed/randperm/setc 为等吞吐；hybrid 按原文语义（4×展开条目）。
偏差登记：vendored simpletransformers 需 2021 版 transformers（无 cp312 轮子），
按上述语义干净重实现；环境与差异记入 run_meta.json。
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MAX_LEN = 128


# ---------------- 编码 ----------------

def encode_raw(raw, tok):
    exs = []
    for s in raw:
        prefix = tok(s["input_text"], add_special_tokens=False)["input_ids"]
        full = tok(s["target_text"], add_special_tokens=False)["input_ids"]
        assert full[:len(prefix)] == prefix, "prefix 必须是 target 的 token 前缀"
        exs.append((prefix, full[len(prefix):]))
    return exs


def build_groups(exs, m, rng):
    idx = list(range(len(exs)))
    rng.shuffle(idx)
    groups = [idx[i:i + m] for i in range(0, len(idx) - m + 1, m)]
    return [[exs[j] for j in mem] for mem in groups]


def ntp_seq(p, a):
    return p + a, [-100] * len(p) + a


def mtp_seq(p, a, eos):
    return p + [eos] * (len(a) - 1) + a[-1:], [-100] * len(p) + a


def pad_batch(seqs, eos):
    """动态 padding（到批内最长，封顶 MAX_LEN）。
    与原实现 pad 到固定 128 数学等价：因果注意下真实 token 不关注后续 pad，
    pad 位置标签为 -100 不进损失。原仓库 preprocess_data_bart 固定 pad 128，
    在 11GB 卡上 vocab 5 万的 logits 张量导致显存压力（登记为等价优化）。"""
    B = len(seqs)
    L = min(max(len(ids) for ids, _ in seqs), MAX_LEN)
    inp = torch.full((B, L), eos, dtype=torch.long)
    lab = torch.full((B, L), -100, dtype=torch.long)
    for b, (ids, labs) in enumerate(seqs):
        n = min(len(ids), L)
        inp[b, :n] = torch.tensor(ids[:n])
        lab[b, :n] = torch.tensor(labs[:n])
    return inp, lab


class UnitDataset(Dataset):
    def __init__(self, units, eos):
        self.units, self.eos = units, eos

    def __len__(self):
        return len(self.units)

    def __getitem__(self, i):
        u = self.units[i]
        return mtp_seq(u[0], u[1], self.eos) if u[2] == "mtp" else ntp_seq(u[0], u[1])


# ---------------- setc / randperm 共用的组级步骤 ----------------

def group_step(model, opt, scaler, groups_ex, group_ids, mode, device, step,
               churn_log, mtp_weight=3.0, matrix_log=None):
    """mode='randperm': 随机双射＋NTP；mode='setc': teacherless 代价+Hungarian+混合。
    group_ids: 每组的全局编号（用于 churn 记录）。
    matrix_log: 若给定列表且 step % matrix_log['every'] == 0，转储全部 4×4 代价矩阵
    （用于最优/次优差距、并列率、结构同质性等离线分析）。"""
    from scipy.optimize import linear_sum_assignment
    eos = model.config.eos_token_id
    G = len(groups_ex)
    m = len(groups_ex[0])

    perms = []
    if mode == "randperm":
        for _ in range(G):
            pi = list(range(m))
            random.shuffle(pi)
            perms.append(pi)
    loss = None

    if mode == "setc":
        # ---- MTP 前向（每 seed 一次，带梯度；m² 代价是同次前向的 gather，向量化）----
        seqs, p_lens, a_lens = [], [], []
        for grp in groups_ex:
            for (p, a) in grp:
                seqs.append(mtp_seq(p, a, eos))
                p_lens.append(len(p))
                a_lens.append(len(a))
        for g in range(G):
            assert len({a_lens[g * m + i] for i in range(m)}) == 1, "组内答案长度需一致"
        L = a_lens[0]
        inp, _ = pad_batch(seqs, eos)
        inp = inp.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(inp).logits
        B = G * m
        # 每个答案槽 t 的预测来自位置 P-1+t（因果 LM 内在 shift）
        p_t = torch.tensor(p_lens, device=device).unsqueeze(1) - 1 + torch.arange(L, device=device)
        b_t = torch.arange(B, device=device).unsqueeze(1).expand(-1, L)
        rows = logits[b_t, p_t].float()                                  # [B, L, V]
        logp = F.log_softmax(rows, dim=-1)
        # 目标 token [G, m(j), L]
        tgt = torch.tensor([[a for (_, a) in grp] for grp in groups_ex], device=device)
        # ll[g,i,j,t] = logp[g*m+i, t, tgt[g,j,t]]（广播高级索引，一次 gather）
        g_i = torch.arange(G, device=device).view(G, 1, 1, 1) * m + \
            torch.arange(m, device=device).view(1, m, 1, 1)              # [G, m(i), 1, 1]
        t_t = torch.arange(L, device=device).view(1, 1, 1, L)
        tgt_b = tgt.unsqueeze(1).expand(G, m, m, L)                      # [G, m(i), m(j), L]
        ll = logp[g_i.expand_as(tgt_b), t_t.expand_as(tgt_b), tgt_b]    # [G, m(i), m(j), L]
        C = -ll.sum(-1)                                                  # [G, m, m]
        # Hungarian 在 detach 代价上选择
        C_np = C.detach().cpu().numpy()
        dump_matrices = (matrix_log is not None
                         and step % matrix_log["every"] == 0)
        for g in range(G):
            r, c = linear_sum_assignment(C_np[g])
            assert list(r) == list(range(m))
            perms.append(list(map(int, c)))
            if churn_log is not None:
                churn_log.append({"step": step, "group": int(group_ids[g]),
                                  "perm": list(map(int, c)),
                                  "cost": float(C_np[g][r, c].sum())})
            if dump_matrices:
                matrix_log["records"].append(
                    {"step": step, "group": int(group_ids[g]),
                     "C": C_np[g].round(4).tolist(), "perm": list(map(int, c))})
        # 被配对条目的 MTP 损失（梯度只流过被选条目；argmin 不可微，EM 式）
        perm_t = torch.tensor(perms, device=device)                      # [G, m]
        sel_ll = ll[torch.arange(G, device=device).view(G, 1, 1),
                    torch.arange(m, device=device).view(1, m, 1),
                    perm_t.view(G, m, 1),
                    torch.arange(L, device=device).view(1, 1, L)]        # [G, m, L]
        mtp_loss = -sel_ll.mean()

    # ---- NTP 前向（被配对对：前缀取 seed i，答案取被配对的 y_j）----
    ntp_seqs = []
    for g in range(G):
        for i in range(m):
            j = perms[g][i]
            p = groups_ex[g][i][0]
            a = groups_ex[g][j][1]
            ntp_seqs.append(ntp_seq(p, a))
    inp2, lab2 = pad_batch(ntp_seqs, eos)
    inp2, lab2 = inp2.to(device), lab2.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
        out = model(inp2, labels=lab2)
    ntp_loss = out.loss.float()

    if mode == "setc":
        loss = ntp_loss + mtp_weight * mtp_loss
    else:
        loss = ntp_loss

    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    return float(loss.detach()), perms


# ---------------- 评价 ----------------

def run_eval(model, tok, probes, raw, task, out_dir, step, device, n_samples, temp, eos):
    model.eval()
    end_id = tok.convert_tokens_to_ids("</a>")
    tok.padding_side = "left"
    texts = [p["input_text"] for p in probes for _ in range(n_samples)]
    outs_flat = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], return_tensors="pt", padding=True).to(device)
            gkw = dict(max_length=MAX_LEN, do_sample=n_samples > 1,
                       pad_token_id=eos, eos_token_id=end_id)
            if n_samples > 1:
                gkw["temperature"] = temp
            o = model.generate(**enc, **gkw)
            for r in range(o.shape[0]):
                outs_flat.append(tok.decode(o[r][enc["input_ids"].shape[1]:],
                                            skip_special_tokens=False))
    items = []
    for k, p in enumerate(probes):
        outs = [p["input_text"] + outs_flat[k * n_samples + r]
                for r in range(n_samples)]
        items.append({"type": p.get("type", "test"),
                      "model_output": outs if n_samples > 1 else outs[0],
                      "target_text": p["target_text"]})
    model.train()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluator import Evaluator
    scores = Evaluator(task, raw).eval_items(items, n_gen=n_samples)
    scores["step"] = step
    with open(os.path.join(out_dir, "scores.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(scores, ensure_ascii=False) + "\n")
    print("[eval]", step, json.dumps(scores, ensure_ascii=False), flush=True)


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["ntp_noseed", "ntp_fixed", "ntp_randperm", "tl_hybrid", "setc_m4"])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--task", choices=["circle", "sibling"], default="circle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--groups_per_step", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--n_probes", type=int, default=256)
    ap.add_argument("--n_samples_probe", type=int, default=1)
    ap.add_argument("--sample_temp", type=float, default=1.0)
    ap.add_argument("--churn_dump", default=None)
    ap.add_argument("--save_final", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    with open(os.path.join(args.data_dir, "vocab.json"), encoding="utf-8") as f:
        new_tokens = json.load(f)
    tok.add_tokens(new_tokens)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.resize_token_embeddings(len(tok))
    eos = tok.eos_token_id
    model.to(device)

    with open(os.path.join(args.data_dir, "train.json"), encoding="utf-8") as f:
        raw = json.load(f)
    exs = encode_raw(raw, tok)

    probes_all = json.load(open(os.path.join(args.data_dir, "test.json"), encoding="utf-8"))
    tr_p = [p for p in probes_all if p.get("type") == "train"][: args.n_probes // 2]
    te_p = [p for p in probes_all if p.get("type") == "test"][: args.n_probes // 2]
    probes = tr_p + te_p

    meta = {"arm": args.arm, "task": args.task, "data_dir": args.data_dir,
            "steps": args.steps, "bs": args.bs, "lr": args.lr, "seed": args.seed,
            "m": args.m, "groups_per_step": args.groups_per_step, "model": args.model,
            "n_train_examples": len(exs), "vocab_size": len(tok),
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0)
            if device == "cuda" else "cpu",
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(args.out, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, ensure_ascii=False), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(args.warmup, 1)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    churn_log = [] if args.arm == "setc_m4" else None
    matrix_log = ({"every": 50, "records": []} if args.arm == "setc_m4" else None)
    step_times = []
    print_every = max(50, args.steps // 40)
    model.train()
    t0_all = time.time()
    step = 0

    if args.arm in ("ntp_randperm", "setc_m4"):
        groups = build_groups(exs, args.m, rng)
        order = []
        while step < args.steps:
            if not order:
                order = list(range(len(groups)))
                rng.shuffle(order)
            pick = order[: args.groups_per_step]
            order = order[args.groups_per_step:]
            batch_groups = [groups[g] for g in pick]
            t0 = time.time()
            mode = "setc" if args.arm == "setc_m4" else "randperm"
            loss, _ = group_step(model, opt, scaler, batch_groups, pick, mode, device,
                                 step, churn_log, matrix_log=matrix_log)
            sched.step()
            step += 1
            if step > 50:
                step_times.append(time.time() - t0)
            if step % print_every == 0:
                print(f"step {step} loss {loss:.4f} "
                      f"sec/step {np.mean(step_times[-100:]):.3f}", flush=True)
            if step % args.eval_every == 0:
                run_eval(model, tok, probes, raw, args.task, args.out, step, device,
                         args.n_samples_probe, args.sample_temp, eos)
    else:
        if args.arm == "tl_hybrid":
            # 消费 hybrid JSON：文件本身已是 1×NTP + 3×MTP 逐例展开
            # （MTP 项 target_text=eos 串、真标签在 target_labels——见 *_hybrid.ipynb）。
            # 若误把该文件当普通数据再展开 1:3，会生成大量"教模型输出 eos"的
            # 单元并淹没 teacherless 目标（首轮 tl_hybrid 失效的根因，已登记）。
            units = []
            if any("target_labels" in s for s in raw):
                for s in raw:
                    prefix = tok(s["input_text"], add_special_tokens=False)["input_ids"]
                    full = tok(s["target_text"], add_special_tokens=False)["input_ids"]
                    labels = tok(s["target_labels"], add_special_tokens=False)["input_ids"]
                    assert full[:len(prefix)] == prefix and labels[:len(prefix)] == prefix
                    kind = "ntp" if s["target_text"] == s["target_labels"] else "mtp"
                    units.append((prefix, labels[len(prefix):], kind))
            else:
                for (p, a) in exs:
                    units.append((p, a, "ntp"))
                    for _ in range(3):
                        units.append((p, a, "mtp"))
        else:
            units = [(p, a, "ntp") for (p, a) in exs]
        loader = DataLoader(UnitDataset(units, eos), batch_size=args.bs, shuffle=True,
                            collate_fn=lambda b: pad_batch(b, eos))
        while step < args.steps:
            for inp, lab in loader:
                if step >= args.steps:
                    break
                t0 = time.time()
                inp, lab = inp.to(device), lab.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=(device == "cuda")):
                    out = model(inp, labels=lab)
                loss = out.loss
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                if step > 50:
                    step_times.append(time.time() - t0)
                if step % print_every == 0:
                    print(f"step {step} loss {loss.item():.4f} "
                          f"sec/step {np.mean(step_times[-100:]):.3f}", flush=True)
                if step % args.eval_every == 0:
                    run_eval(model, tok, probes, raw, args.task, args.out, step, device,
                             args.n_samples_probe, args.sample_temp, eos)

    thr = {"sec_per_step_mean": float(np.mean(step_times)),
           "sec_per_step_p50": float(np.median(step_times)),
           "total_hours": (time.time() - t0_all) / 3600, "steps": args.steps}
    with open(os.path.join(args.out, "throughput.json"), "w", encoding="utf-8") as f:
        json.dump(thr, f, indent=2)
    print("throughput", json.dumps(thr), flush=True)
    if churn_log is not None and args.churn_dump:
        with open(args.churn_dump, "w", encoding="utf-8") as f:
            for rec in churn_log:
                f.write(json.dumps(rec) + "\n")
    if matrix_log is not None and args.churn_dump:
        p, _ = os.path.splitext(args.churn_dump)
        with open(p + "_matrices.jsonl", "w", encoding="utf-8") as f:
            for rec in matrix_log["records"]:
                f.write(json.dumps(rec) + "\n")
    if args.save_final:
        model.save_pretrained(os.path.join(args.out, "final_model"))
        tok.save_pretrained(os.path.join(args.out, "final_model"))
    run_eval(model, tok, probes, raw, args.task, args.out, args.steps, device,
             args.n_samples_probe, args.sample_temp, eos)


if __name__ == "__main__":
    main()
