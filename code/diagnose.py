"""P6 WP3: 诊断脚本。

1. churn（G2 诊断）：setc 的 churn_dump JSONL → 每组相邻两次访问的 π* 的 Cayley 距离
   （= m − 置换循环数），按步数窗口聚合；同时统计恒等置换占比（冻结信号）。
2. I(z;[y]) 代理：对已见/未见 hash 各 K 个，每 hash 采 n 样本（temp=1），
   输出按任务等价类归并 → H([y]|z) 的插件估计，I 代理 = H([y]) − mean H([y]|z)。
   注意：n=16 的插件估计下偏，作代理而非无偏估计。
3. 覆盖曲线：每 hash 累计不同有效等价类数 vs 采样数（circle），以及池化覆盖。
4. teacherless 一次读取：hybrid/setc 模型在 MTP 输入（prefix+pads）上并行 argmax，
   报告一次读取的有效率（对照 AR greedy 有效率）。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluator import Evaluator, circle_canonicalize, complete  # noqa: E402


# ---------- churn ----------

def cycles(p):
    seen = set()
    c = 0
    for i in range(len(p)):
        if i in seen:
            continue
        c += 1
        j = i
        while j not in seen:
            seen.add(j)
            j = p[j]
    return c


def cayley(p1, p2):
    """σ = p2 ∘ p1^{-1} 的 Cayley 距离 = m − cycles(σ)。"""
    m = len(p1)
    inv1 = [0] * m
    for i, v in enumerate(p1):
        inv1[v] = i
    sigma = [p2[inv1[i]] for i in range(m)]
    return m - cycles(sigma)


def churn_analysis(dump_path, m=4, window=200):
    by_group = {}
    with open(dump_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_group.setdefault(rec["group"], []).append((rec["step"], rec["perm"]))
    dists, steps_all = [], []
    ident_frac_win = {}
    for g, visits in by_group.items():
        visits.sort()
        for (s1, p1), (s2, p2) in zip(visits, visits[1:]):
            dists.append(cayley(p1, p2))
            steps_all.append(s2)
        for s, p in visits:
            w = s // window
            ident_frac_win.setdefault(w, []).append(1.0 if p == list(range(m)) else 0.0)
    order = np.argsort(steps_all)
    dists_o = np.array(dists)[order]
    steps_o = np.array(steps_all)[order]
    traj = []
    for lo in range(0, len(dists_o), 500):
        sel = dists_o[lo:lo + 500]
        traj.append({"steps_range": [int(steps_o[lo]) if lo < len(steps_o) else None,
                                     int(steps_o[min(lo + 499, len(steps_o) - 1)])],
                     "mean_cayley": float(sel.mean()), "n": int(len(sel))})
    ident = [{"window_step": w * window, "identity_frac": float(np.mean(v)), "n": len(v)}
             for w, v in sorted(ident_frac_win.items())]
    return {"n_groups": len(by_group),
            "mean_visits_per_group": float(np.mean([len(v) for v in by_group.values()])),
            "mean_cayley": float(np.mean(dists)) if dists else None,
            "traj": traj, "identity_frac": ident}


# ---------- 采样与 I 代理 ----------

def sample_hashes(model, tok, hashes, task, n, temp, eos, device, max_len=64):
    """每 hash 采 n 个输出，返回 {hash: [ans_str]}。"""
    end_id = tok.convert_tokens_to_ids("</a>")
    tok.padding_side = "left"
    out = {}
    model.eval()
    with torch.no_grad():
        for h in hashes:
            enc = tok([h] * n, return_tensors="pt", padding=True).to(device)
            gkw = dict(max_length=max_len, do_sample=True, temperature=temp,
                       pad_token_id=eos, eos_token_id=end_id)
            if temp == 1.0:
                gkw.pop("temperature", None)
            o = model.generate(**enc, **gkw)
            outs = []
            for r in range(n):
                s = tok.decode(o[r][enc["input_ids"].shape[1]:], skip_special_tokens=False)
                outs.append(s)
            out[h] = outs
    return out


def key_of(task, ans):
    if task == "circle":
        a = ans.split("<q>")[1] if "<q>" in ans else ans
        return circle_canonicalize(a)
    return a if (a := ans.split("</a>")[0]) else None


def entropy_probe(samples_by_hash, task):
    """返回 (H_pooled, mean_H_cond, I_proxy_bits, per_hash_H)。仅计可解析输出。"""
    def cls_counts(lst):
        c = {}
        for s in lst:
            k = key_of(task, s)
            if k is None:
                continue
            c[k] = c.get(k, 0) + 1
        return c

    pooled = {}
    cond_H = []
    per = []
    for h, lst in samples_by_hash.items():
        c = cls_counts(lst)
        if not c:
            continue
        n = sum(c.values())
        p = np.array(list(c.values())) / n
        H = float(-(p * np.log2(p)).sum())
        cond_H.append(H)
        per.append({"hash": h[:14], "H_bits": H, "n_valid": n, "n_classes": len(c)})
        for k, v in c.items():
            pooled[k] = pooled.get(k, 0) + v
    if not pooled:
        return 0.0, 0.0, 0.0, []
    n = sum(pooled.values())
    p = np.array(list(pooled.values())) / n
    H_pool = float(-(p * np.log2(p)).sum())
    mean_H = float(np.mean(cond_H)) if cond_H else 0.0
    return H_pool, mean_H, H_pool - mean_H, per


def coverage_curve(samples_by_hash, task):
    """池化累计覆盖：随每 hash 样本预算增加，累计不同有效等价类数。"""
    per_n = {}
    for h, lst in samples_by_hash.items():
        seen = set()
        for i, s in enumerate(lst):
            k = key_of(task, s)
            if k is not None:
                seen.add(k)
            per_n[i + 1] = per_n.get(i + 1, set()) | seen
    return [{"n_per_hash": n, "unique_valid_pooled": len(v)}
            for n, v in sorted(per_n.items())]


# ---------- teacherless 一次读取 ----------

def teacherless_readout(model, tok, probes, task, L_ans, eos, device, bs=64):
    """MTP 输入（prefix + eos×(L-1) + </a>）并行 argmax 读取答案槽。"""
    end_id = tok.convert_tokens_to_ids("</a>")
    outs = []
    with torch.no_grad():
        for i in range(0, len(probes), bs):
            chunk = probes[i:i + bs]
            enc = [tok(p["input_text"], add_special_tokens=False)["input_ids"]
                   for p in chunk]
            inp = []
            for p_ids in enc:
                inp.append(p_ids + [eos] * (L_ans - 1) + [end_id])
            B = len(inp)
            t = torch.full((B, max(len(x) for x in inp)), eos, dtype=torch.long)
            for b, x in enumerate(inp):
                t[b, :len(x)] = torch.tensor(x)
            t = t.to(device)
            logits = model(t).logits
            for b, p_ids in enumerate(enc):
                P = len(p_ids)
                ids = logits[b, P - 1:P - 1 + L_ans, :].argmax(-1).tolist()
                outs.append(tok.decode(ids, skip_special_tokens=False))
    # 组装回 pred（带前缀以便 evaluator 解析）
    items = []
    for p, ans in zip(probes, outs):
        prefix = p["input_text"]
        items.append({"type": p.get("type", "test"),
                      "model_output": prefix + ans + "</a>",
                      "target_text": p["target_text"]})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["churn", "seeds", "readout"], required=True)
    ap.add_argument("--churn_dump")
    ap.add_argument("--model_dir", help="final_model 目录")
    ap.add_argument("--data_dir")
    ap.add_argument("--task", choices=["circle", "sibling"], default="circle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=32, help="已见/未见 hash 各多少")
    ap.add_argument("--n", type=int, default=16, help="每 hash 采样数")
    ap.add_argument("--temp", type=float, default=1.0)
    args = ap.parse_args()

    if args.mode == "churn":
        res = churn_analysis(args.churn_dump)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(json.dumps({k: v for k, v in res.items() if k not in ("traj", "identity_frac")},
                         indent=2))
        return

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(args.model_dir)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    eos = tok.eos_token_id

    with open(os.path.join(args.data_dir, "train.json"), encoding="utf-8") as f:
        raw = json.load(f)
    with open(os.path.join(args.data_dir, "test.json"), encoding="utf-8") as f:
        probes_all = json.load(f)

    if args.mode == "seeds":
        rng = np.random.default_rng(7)
        seen_hashes = [raw[i]["input_text"] for i in
                       rng.choice(len(raw), args.K, replace=False)]
        unseen_hashes = [p["input_text"] for p in probes_all
                         if p.get("type") == "test"][: args.K]
        res = {"K": args.K, "n": args.n, "temp": args.temp}
        for name, hs in [("seen_seed", seen_hashes), ("unseen_seed", unseen_hashes)]:
            samples = sample_hashes(model, tok, hs, args.task, args.n, args.temp,
                                    eos, device)
            H, Hc, I, per = entropy_probe(samples, args.task)
            cov = coverage_curve(samples, args.task)
            res[name] = {"H_pooled_bits": H, "mean_H_cond_bits": Hc,
                         "I_proxy_bits": I, "coverage": cov[-1] if cov else None,
                         "per_hash": per}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        s = {k: {kk: vv for kk, vv in v.items() if kk != "per_hash"}
             for k, v in res.items() if isinstance(v, dict)}
        print(json.dumps(s, indent=2))
        return

    if args.mode == "readout":
        # L_ans 从训练数据统计（任务内等长）
        full = tok(raw[0]["target_text"], add_special_tokens=False)["input_ids"]
        pref = tok(raw[0]["input_text"], add_special_tokens=False)["input_ids"]
        L_ans = len(full) - len(pref)
        te = [p for p in probes_all if p.get("type") == "test"][:256]
        tr = [p for p in probes_all if p.get("type") == "train"][:128]
        res = {"L_ans": L_ans}
        ev = Evaluator(args.task, raw)
        for name, ps in [("train", tr), ("test", te)]:
            items = teacherless_readout(model, tok, ps, args.task, L_ans, eos, device)
            sc = ev.eval_items(items)
            res[name] = sc
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
