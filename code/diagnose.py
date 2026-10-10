"""P6 WP3: 诊断脚本。

1. churn（G2 诊断）：setc 的 churn_dump JSONL → 每组相邻两次访问的 π* 的 Cayley 距离
   （= m − 置换循环数），按步数窗口聚合；同时统计恒等置换占比（冻结信号）。
2. I(z;[y]) 代理：对已见/未见 hash 各 K 个，每 hash 采 n 样本（temp=1），
   有效输出按任务等价类归并 → H([y]|z) 的插件估计，I 代理 = H([y]) − mean H([y]|z)。
   注意：插件估计受样本量封顶（H([y]|z) ≤ log2 n，H([y]) ≤ log2 Kn）；circle 这类
   几乎每个样本各成一类的任务上，即使 z 与 y 独立，I 代理也≈log2 K。因此同时报告
   置换零分布（把样本随机重新分给各 hash 后的 I 代理）——只有超出零分布的部分才有意义。
3. 覆盖曲线：每 hash 累计不同有效等价类数 vs 采样数（circle），以及池化覆盖。
4. teacherless 一次读取：hybrid/setc 模型在 MTP 输入（prefix+pads）上并行 argmax，
   报告一次读取的有效率（对照 AR greedy 有效率）。
5. mi：基于模型似然的 I(z;y) 估计。z 在 K 个 hash 上均匀，y ~ p(·|z) 采样，
   Î = mean_y [log p(y|z_k) − log (1/K) Σ_k' p(y|z_k')]（比特；上界 log2 K）。
   用的是精确条件似然而非等价类计数，不受上述插件偏差影响。
6. agree：teacherless 代价（训练用的廉价代理）与 AR 代价（测试时真正生成所用的似然）
   各自 Hungarian 选出的配对是否一致——C01 的实质问题（加速比≈m 是前向次数的算术）。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluator import Evaluator, circle_canonicalize, complete  # noqa: E402
from train import assign, build_index_groups, encode_raw, truncate_answer  # noqa: E402


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


def churn_analysis(dump_path, window=200):
    """--regroup 运行中组编号只在本轮内唯一，按成员集合区分组（跨轮成员不同则不计 churn）。
    tl_fixed/tl_randperm 记录含影子 argmin：另报告"被施加配对 == 模型 argmin"的比例。"""
    by_group = {}
    agree_win = {}
    with open(dump_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = tuple(rec["members"]) if "members" in rec else rec["group"]
            by_group.setdefault(key, []).append((rec["step"], rec["perm"]))
            if "argmin" in rec:
                agree_win.setdefault(rec["step"] // window, []).append(
                    1.0 if rec["argmin"] == rec["perm"] else 0.0)
    dists, steps_all = [], []
    ident_frac_win = {}
    for g, visits in by_group.items():
        visits.sort()
        for (s1, p1), (s2, p2) in zip(visits, visits[1:]):
            dists.append(cayley(p1, p2))
            steps_all.append(s2)
        for s, p in visits:
            w = s // window
            ident_frac_win.setdefault(w, []).append(1.0 if p == list(range(len(p))) else 0.0)
    if not dists:
        return {"n_groups": len(by_group), "mean_cayley": None, "traj": [],
                "identity_frac": [], "note": "无同一组的重复访问（--regroup？）"}
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
    res = {"n_groups": len(by_group),
           "mean_visits_per_group": float(np.mean([len(v) for v in by_group.values()])),
           "mean_cayley": float(np.mean(dists)) if dists else None,
           "traj": traj, "identity_frac": ident}
    if agree_win:
        res["argmin_agree_frac"] = [{"window_step": w * window, "frac": float(np.mean(v)),
                                     "n": len(v)} for w, v in sorted(agree_win.items())]
    return res


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
                outs.append(truncate_answer(s))  # 去掉批内提前停止行的尾部 pad
            out[h] = outs
    return out


def key_of(task, ans, ev):
    """有效输出的等价类键；无效或不可解析返回 None。ans 为 <q> 之后的续写。"""
    a = ans.split("<q>")[1] if "<q>" in ans else ans
    a = a.split("</a>")[0]
    if not ev.validity("<q>" + a + "</a>"):
        return None
    return circle_canonicalize(a) if task == "circle" else a


def entropy_probe(samples_by_hash, task, ev):
    """返回 (H_pooled, mean_H_cond, I_proxy_bits, per_hash_H)。仅计有效输出。"""
    def cls_counts(lst):
        c = {}
        for s in lst:
            k = key_of(task, s, ev)
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


def permutation_null(samples_by_hash, task, ev, n_perm=20, seed=0):
    """把全部样本随机重新分给各 hash（保持每 hash 样本数），重算 I 代理。
    零分布刻画"z 与 y 独立"时插件估计仍会给出的数值。"""
    rng = np.random.default_rng(seed)
    hs = list(samples_by_hash)
    sizes = [len(samples_by_hash[h]) for h in hs]
    pool = [s for h in hs for s in samples_by_hash[h]]
    vals = []
    for _ in range(n_perm):
        perm = rng.permutation(len(pool))
        shuf, k = {}, 0
        for h, n in zip(hs, sizes):
            shuf[h] = [pool[i] for i in perm[k:k + n]]
            k += n
        vals.append(entropy_probe(shuf, task, ev)[2])
    return {"I_null_mean_bits": float(np.mean(vals)),
            "I_null_p95_bits": float(np.percentile(vals, 95)), "n_perm": n_perm}


def coverage_curve(samples_by_hash, task, ev):
    """池化累计覆盖：随每 hash 样本预算增加，累计不同有效等价类数。"""
    per_n = {}
    for h, lst in samples_by_hash.items():
        seen = set()
        for i, s in enumerate(lst):
            k = key_of(task, s, ev)
            if k is not None:
                seen.add(k)
            per_n[i + 1] = per_n.get(i + 1, set()) | seen
    return [{"n_per_hash": n, "unique_valid_pooled": len(v)}
            for n, v in sorted(per_n.items())]


# ---------- 基于似然的 I(z;y)、代价一致性 ----------

def seq_logprobs(model, pairs, device, pad_id, bs=64):
    """pairs: [(prefix_ids, ans_ids)] → 每条 Σ_t log p(ans_t | prefix, ans_<t)（teacher forcing）。"""
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            L = max(len(p) + len(a) for p, a in chunk)
            inp = torch.full((len(chunk), L), pad_id, dtype=torch.long)
            for b, (p, a) in enumerate(chunk):
                inp[b, :len(p) + len(a)] = torch.tensor(p + a)
            logits = model(inp.to(device)).logits.float()
            for b, (p, a) in enumerate(chunk):
                pos = torch.arange(len(p) - 1, len(p) - 1 + len(a), device=device)
                lp = F.log_softmax(logits[b, pos], dim=-1)
                out.append(float(lp[torch.arange(len(a), device=device),
                                    torch.tensor(a, device=device)].sum()))
    return out


def sample_ids(model, tok, prompt_ids, n, temp, eos, device, max_new=64):
    """对单个 prompt 采 n 条，返回答案 token id 列表（截到首个 </a>，含；无 </a> 的丢弃）。"""
    end_id = tok.convert_tokens_to_ids("</a>")
    inp = torch.tensor([prompt_ids] * n, device=device)
    with torch.no_grad():
        o = model.generate(inp, attention_mask=torch.ones_like(inp), do_sample=True,
                           temperature=temp, max_new_tokens=max_new,
                           pad_token_id=eos, eos_token_id=end_id)
    outs = []
    for r in range(n):
        ids = o[r, len(prompt_ids):].tolist()
        if end_id in ids:
            outs.append(ids[:ids.index(end_id) + 1])
    return outs


def mi_estimate(model, tok, prompts, n, temp, eos, device):
    """prompts: K 个 hash 的 prefix ids。返回 Î(z;y)（比特）及有效样本数。
    Î = mean_{k, y~p(·|z_k)} [log p(y|z_k) − logsumexp_k' log p(y|z_k') + log K]。"""
    K = len(prompts)
    samples = [(k, y) for k, p in enumerate(prompts)
               for y in sample_ids(model, tok, p, n, temp, eos, device)]
    if not samples:
        return {"I_bits": None, "n_samples": 0}
    pairs = [(prompts[k2], y) for (_, y) in samples for k2 in range(K)]
    lp = torch.tensor(seq_logprobs(model, pairs, device, eos)).view(len(samples), K)
    own = lp[torch.arange(len(samples)), torch.tensor([k for k, _ in samples])]
    pmi = own - (torch.logsumexp(lp, dim=1) - np.log(K))
    return {"I_bits": float(pmi.mean() / np.log(2)), "upper_bound_bits": float(np.log2(K)),
            "pmi_se_bits": float(pmi.std() / np.sqrt(len(samples)) / np.log(2)),
            "n_samples": len(samples), "K": K}


def teacherless_costs(model, groups, eos, end_id, device):
    """groups: [[(p, a)]*m] → [G, m, m]，C[g,i,j] = −Σ_t log q_t(a_j,t | p_i, dummy)。"""
    G, m = len(groups), len(groups[0])
    C = np.zeros((G, m, m))
    with torch.no_grad():
        for g, grp in enumerate(groups):
            L = len(grp[0][1])
            seqs = [p + [eos] * (L - 1) + [end_id] for p, _ in grp]
            T = max(len(s) for s in seqs)
            inp = torch.full((m, T), eos, dtype=torch.long)
            for i, s in enumerate(seqs):
                inp[i, :len(s)] = torch.tensor(s)
            logits = model(inp.to(device)).logits.float()
            for i, (p, _) in enumerate(grp):
                lp = F.log_softmax(logits[i, len(p) - 1:len(p) - 1 + L], dim=-1)
                for j, (_, a) in enumerate(grp):
                    C[g, i, j] = -float(lp[torch.arange(L, device=device),
                                           torch.tensor(a, device=device)].sum())
    return C


def assignment_agreement(model, groups, eos, end_id, device):
    """teacherless 与 AR 代价各自的 Hungarian 配对的一致性及 AR 意义下的遗憾。"""
    G, m = len(groups), len(groups[0])
    C_tl = teacherless_costs(model, groups, eos, end_id, device)
    pairs = [(grp[i][0], grp[j][1]) for grp in groups for i in range(m) for j in range(m)]
    C_ar = -np.array(seq_logprobs(model, pairs, device, eos)).reshape(G, m, m)
    same, dist, regret, rho = [], [], [], []
    for g in range(G):
        p_tl, p_ar = assign(C_tl[g]), assign(C_ar[g])
        same.append(p_tl == p_ar)
        dist.append(cayley(p_tl, p_ar))
        regret.append(C_ar[g][range(m), p_tl].sum() - C_ar[g][range(m), p_ar].sum())
        a, b = C_tl[g].ravel(), C_ar[g].ravel()
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        rho.append(float(np.corrcoef(ra, rb)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan)
    return {"G": G, "m": m, "exact_agree_frac": float(np.mean(same)),
            "mean_cayley_tl_vs_ar": float(np.mean(dist)),
            "chance_cayley": float(m - sum(1.0 / k for k in range(1, m + 1))),
            "mean_ar_regret_nats": float(np.mean(regret)),
            "mean_spearman_cost": float(np.nanmean(rho)),
            "perms_tl": [assign(C_tl[g]) for g in range(min(G, 8))]}


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
        # L_ans 含 </a> 槽：读取正确时 ans 已以 </a> 结尾，不能再直接追加
        # （否则 </a> 出现两次 → validity 恒为 0）。截到首个 </a>，缺失才补。
        items.append({"type": p.get("type", "test"),
                      "model_output": prefix + truncate_answer(ans + "</a>"),
                      "target_text": p["target_text"]})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["churn", "seeds", "readout", "mi", "agree"],
                    required=True)
    ap.add_argument("--churn_dump")
    ap.add_argument("--model_dir", help="final_model 目录")
    ap.add_argument("--data_dir")
    ap.add_argument("--task", choices=["circle", "sibling"], default="circle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=32, help="已见/未见 hash 各多少")
    ap.add_argument("--n", type=int, default=16, help="每 hash 采样数")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--n_perm", type=int, default=20, help="seeds: 置换零分布次数")
    ap.add_argument("--G", type=int, default=64, help="agree: 评估多少个组")
    ap.add_argument("--m", type=int, default=4, help="agree: 组大小")
    ap.add_argument("--train_seed", type=int, default=42,
                    help="agree: 用训练 --seed 复原训练时的分组（setc 冻结配对即定义在其上）")
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
        ev = Evaluator(args.task, raw)
        for name, hs in [("seen_seed", seen_hashes), ("unseen_seed", unseen_hashes)]:
            samples = sample_hashes(model, tok, hs, args.task, args.n, args.temp,
                                    eos, device)
            H, Hc, I, per = entropy_probe(samples, args.task, ev)
            null = permutation_null(samples, args.task, ev, n_perm=args.n_perm)
            cov = coverage_curve(samples, args.task, ev)
            res[name] = {"H_pooled_bits": H, "mean_H_cond_bits": Hc,
                         "I_proxy_bits": I, **null,
                         "I_proxy_minus_null_bits": I - null["I_null_mean_bits"],
                         "coverage": cov[-1] if cov else None,
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
        return

    if args.mode == "mi":
        rng = np.random.default_rng(7)
        enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
        seen_p = [enc(raw[i]["input_text"]) for i in rng.choice(len(raw), args.K, replace=False)]
        unseen_p = [enc(p["input_text"]) for p in probes_all if p.get("type") == "test"][: args.K]
        res = {"K": args.K, "n": args.n, "temp": args.temp,
               "seen_seed": mi_estimate(model, tok, seen_p, args.n, args.temp, eos, device),
               "unseen_seed": mi_estimate(model, tok, unseen_p, args.n, args.temp, eos, device)}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))
        return

    if args.mode == "agree":
        exs = encode_raw(raw, tok)
        idx_groups = build_index_groups(len(exs), args.m, np.random.default_rng(args.train_seed))
        groups = [[exs[j] for j in mem] for mem in idx_groups[: args.G]]
        res = assignment_agreement(model, groups, eos, tok.convert_tokens_to_ids("</a>"),
                                   device)
        res["group_ids"] = list(range(min(args.G, len(idx_groups))))
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
