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
import glob
import gzip
import itertools
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluator import Evaluator, circle_canonicalize  # noqa: E402
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


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") \
        else open(path, encoding="utf-8")


def cycle_type(p):
    seen, lens = set(), []
    for s in range(len(p)):
        if s in seen:
            continue
        k, j = 0, s
        while j not in seen:
            seen.add(j)
            j = p[j]
            k += 1
        lens.append(k)
    return tuple(sorted(lens, reverse=True))


def read_churn(dump_path):
    """→ {组键: [(step, perm)]}（按 step 排序）与影子 argmin 记录。
    --regroup 运行中组编号只在本轮内唯一，按成员集合区分组。"""
    by_group, argmin = {}, []
    with open_text(dump_path) as f:
        for line in f:
            rec = json.loads(line)
            key = tuple(rec["members"]) if "members" in rec else rec["group"]
            by_group.setdefault(key, []).append((rec["step"], rec["perm"]))
            if "argmin" in rec:
                argmin.append((rec["step"], rec["argmin"] == rec["perm"]))
    for v in by_group.values():
        v.sort()
    return by_group, argmin


def churn_analysis(dump_path, window=200, struct_keys=None, commit_bin=1000):
    """配对动力学。除相邻访问 Cayley 距离与恒等占比外，另报告（评审第 3–4 轮）：
    - final_perm_stats：各组最终配对的循环型分布、不动点占比、恒等占比（对照均匀 S_m）；
    - p_eq_final：各窗口内访问的配对 == 该组最终配对 的比例（富者愈富：早期即应 > 1/m!）；
    - commitment：各组"此后每次访问都等于最终配对"的最早步（承诺时间）分布；
    - struct_change_traj（给定 struct_keys 时）：只把"某个 seed 被分到的 canonical 结构变了"计为变化
      ——目标结构相同的两个条目互换不算换了生成方案（circle 1 万个目标只有 8,859 个结构）。
    struct_keys: {组键: [组员 0..m-1 的目标结构键]}。
    tl_fixed/tl_randperm/锁定分支的记录含影子 argmin：另报告"施加配对 == 模型 argmin"的比例。"""
    by_group, argmin = read_churn(dump_path)
    dists, steps_all, schg = [], [], []
    ident_frac_win, eqf_win = {}, {}
    commit, final = [], {}
    for g, visits in by_group.items():
        fin = visits[-1][1]
        final[g] = fin
        for (s1, p1), (s2, p2) in zip(visits, visits[1:]):
            dists.append(cayley(p1, p2))
            steps_all.append(s2)
            if struct_keys is not None and g in struct_keys:
                k = struct_keys[g]
                schg.append((s2, any(k[p1[i]] != k[p2[i]] for i in range(len(p1)))))
        for s, p in visits:
            w = s // window
            ident_frac_win.setdefault(w, []).append(1.0 if p == list(range(len(p))) else 0.0)
            eqf_win.setdefault(w, []).append(1.0 if p == fin else 0.0)
        t_commit = visits[-1][0]
        for s, p in reversed(visits):
            if p != fin:
                break
            t_commit = s
        commit.append(t_commit)
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
    m = len(next(iter(final.values())))
    ctypes = {}
    for p in final.values():
        ctypes[cycle_type(p)] = ctypes.get(cycle_type(p), 0) + 1
    commit = np.array(commit)
    hist = {}
    for c in commit:
        hist[int(c // commit_bin * commit_bin)] = hist.get(int(c // commit_bin * commit_bin), 0) + 1
    res = {"n_groups": len(by_group),
           "mean_visits_per_group": float(np.mean([len(v) for v in by_group.values()])),
           "mean_cayley": float(np.mean(dists)) if dists else None,
           "chance_cayley": float(m - sum(1.0 / k for k in range(1, m + 1))),
           "traj": traj, "identity_frac": ident,
           "final_perm_stats": {
               "fixed_point_frac": float(np.mean([sum(p[i] == i for i in range(m)) / m
                                                  for p in final.values()])),
               "identity_share": float(np.mean([p == list(range(m)) for p in final.values()])),
               "cycle_types": {"+".join(map(str, k)): v / len(final)
                               for k, v in sorted(ctypes.items())},
               "uniform_fixed_point_frac": 1.0 / m},
           "p_eq_final": [{"window_step": w * window, "frac": float(np.mean(v)), "n": len(v)}
                          for w, v in sorted(eqf_win.items())],
           "chance_p_eq_final": 1.0 / float(np.prod(range(1, m + 1))),
           "commitment": {"q10": float(np.quantile(commit, 0.1)),
                          "median": float(np.median(commit)),
                          "q90": float(np.quantile(commit, 0.9)),
                          "hist": dict(sorted(hist.items()))}}
    if schg:
        schg.sort()
        st = np.array([c for _, c in schg], dtype=float)
        ss = np.array([s for s, _ in schg])
        res["struct_change_traj"] = [
            {"steps_range": [int(ss[lo]), int(ss[min(lo + 499, len(ss) - 1)])],
             "frac": float(st[lo:lo + 500].mean()), "n": int(len(st[lo:lo + 500]))}
            for lo in range(0, len(st), 500)]
        res["perm_change_but_same_structures"] = float(np.mean(
            [(d > 0) and not c for d, (_, c) in zip(np.array(dists)[order], schg)]))
    if argmin:
        aw = {}
        for s, eq in argmin:
            aw.setdefault(s // window, []).append(1.0 if eq else 0.0)
        res["argmin_agree_frac"] = [{"window_step": w * window, "frac": float(np.mean(v)),
                                     "n": len(v)} for w, v in sorted(aw.items())]
    return res


def churn_struct_keys(data_dir, task, train_seed, m, by_group):
    """组键 → 组员目标的结构键（circle 为 canonical；sibling 为答案串）。"""
    with open(os.path.join(data_dir, "train.json"), encoding="utf-8") as f:
        raw = json.load(f)
    def key(s):
        a = s["target_text"].split("<q>")[1].split("</a>")[0]
        return circle_canonicalize(a) if task == "circle" else a
    keys = [key(s) for s in raw]
    groups = build_index_groups(len(raw), m, np.random.default_rng(train_seed))
    out = {}
    for g in by_group:
        mem = list(g) if isinstance(g, tuple) else groups[g]
        out[g] = [keys[j] for j in mem]
    return out


def export_pairing(dump_path, n, train_seed, m, at_step=None, base_pairing=None):
    """churn → 全局 seed→target 配对（训练样本编号）：各组在 at_step（缺省=最后一次访问）
    或之前最近一次访问的配对。未访问的组保持名义。base_pairing：若该运行本身用了
    --pairing_file，传入其 R 以复合（target = R[组员 perm(i)]）。"""
    by_group, _ = read_churn(dump_path)
    groups = build_index_groups(n, m, np.random.default_rng(train_seed))
    R = base_pairing if base_pairing is not None else list(range(n))
    cmap = {s: R[s] for s in range(n)}
    n_groups = 0
    for g, visits in by_group.items():
        if isinstance(g, tuple):
            raise ValueError("--regroup 运行没有持久组，不能导出组内配对")
        vs = [v for v in visits if at_step is None or v[0] <= at_step]
        if not vs:
            continue
        perm = vs[-1][1]
        mem = groups[g]
        for i in range(m):
            cmap[mem[i]] = R[mem[perm[i]]]
        n_groups += 1
    return cmap, n_groups


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


def tl_cost_tensor(model, groups, eos, end_id, device, bs=64):
    """逐位置 teacherless 代价 [G, m, m, L]：c[g,i,j,t] = −log q_t(a_j,t | p_i, dummy)。
    每个 seed 一次前向（批量），m² 个代价是同一前向的 gather——与训练中 setc 的代价一致。"""
    G, m = len(groups), len(groups[0])
    L = len(groups[0][0][1])
    assert all(len(a) == L for grp in groups for _, a in grp), "答案须等长"
    tgt = torch.tensor([[a for _, a in grp] for grp in groups], device=device)   # [G, m, L]
    flat = [(g, i, p) for g, grp in enumerate(groups) for i, (p, _) in enumerate(grp)]
    out = np.zeros((G, m, m, L))
    with torch.no_grad():
        for c0 in range(0, len(flat), bs):
            chunk = flat[c0:c0 + bs]
            seqs = [p + [eos] * (L - 1) + [end_id] for _, _, p in chunk]
            inp = torch.full((len(seqs), max(len(s) for s in seqs)), eos, dtype=torch.long)
            for b, s in enumerate(seqs):
                inp[b, :len(s)] = torch.tensor(s)
            logits = model(inp.to(device)).logits.float()
            for b, (g, i, p) in enumerate(chunk):
                lp = F.log_softmax(logits[b, len(p) - 1:len(p) - 1 + L], dim=-1)   # [L, V]
                out[g, i] = -lp.gather(1, tgt[g].T).T.cpu().numpy()             # [m, L]
    return out


def ar_costs(model, groups, eos, device):
    """AR（teacher-forcing）代价 [G, m, m]：C[g,i,j] = −log p(a_j | p_i)。G·m² 条序列。"""
    G, m = len(groups), len(groups[0])
    pairs = [(grp[i][0], grp[j][1]) for grp in groups for i in range(m) for j in range(m)]
    return -np.array(seq_logprobs(model, pairs, device, eos)).reshape(G, m, m)


def double_center(M):
    """去掉行、列主效应后的交互项：只有它影响双射的选择（C+a_i+b_j 不改变最优配对）。"""
    return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()


def own_perms_from_pairing(groups_idx, ref):
    """组内自身配对：ref 为全局 seed→target；目标不在组内的组记 None。"""
    out = []
    for mem in groups_idx:
        pos = {e: k for k, e in enumerate(mem)}
        p = [pos.get(ref[e]) for e in mem]
        out.append(p if None not in p and sorted(p) == list(range(len(mem))) else None)
    return out


def _perm_table(C, perms):
    idx = np.arange(C.shape[0])
    return np.array([C[idx, list(p)].sum() for p in perms])


def gap_stats(C, reps, own, L):
    """单一代价类型的最优-次优间隔与自身配对间隔（按噪声尺度与 token 数归一）。
    C: [G, m, m]（eval 模式）；reps: [R, G, m, m]（dropout 复本，可空）；own: 各组自身排列或 None。"""
    G, m = C.shape[0], C.shape[1]
    if m > 6:
        return {"note": "m>6 不穷举排列"}
    P = list(itertools.permutations(range(m)))
    gaps, zs, robust, ties_exact, ties_noise = [], [], [], [], []
    own_best, own_margin, own_z, own_rank = [], [], [], []
    for g in range(G):
        pc = _perm_table(C[g], P)
        o = np.argsort(pc, kind="stable")
        gap = float(pc[o[1]] - pc[o[0]])
        gaps.append(gap)
        ties_exact.append(gap == 0.0)
        if len(reps):
            d = np.array([_perm_table(r[g], [P[o[1]]])[0] - _perm_table(r[g], [P[o[0]]])[0]
                          for r in reps])
            sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
            eps = float(np.max(np.abs(np.asarray(reps)[:, g] - C[g])))
            zs.append(gap / sd if sd > 0 else float("inf"))
            ties_noise.append(gap < sd)
            robust.append(gap > 2 * m * eps)   # 充分条件：每元扰动 ≤ε 时最优排列不变
        if own[g] is not None:
            k = P.index(tuple(own[g]))
            others = np.delete(pc, k)
            mg = float(others.min() - pc[k])          # >0：自身配对是唯一最优，间隔 mg
            own_margin.append(mg)
            own_best.append(mg > 0)
            own_rank.append(int(np.where(o == k)[0][0]))
            if len(reps):
                d = np.array([np.delete(_perm_table(r[g], P), k).min() - _perm_table(r[g], P)[k]
                              for r in reps])
                sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
                own_z.append(mg / sd if sd > 0 else float("inf"))
    q = lambda a: {"q10": float(np.quantile(a, .1)), "median": float(np.median(a)),  # noqa: E731
                   "q90": float(np.quantile(a, .9))} if len(a) else None
    res = {"gap_nats": q(gaps), "gap_per_token": q(np.array(gaps) / (m * L)),
           "exact_tie_frac": float(np.mean(ties_exact))}
    if zs:
        res.update({"gap_over_dropout_sd": q([z for z in zs if np.isfinite(z)]),
                    "noise_tie_frac": float(np.mean(ties_noise)),
                    "robust_frac_2m_eps": float(np.mean(robust))})
    if own_margin:
        res.update({"own_is_best_frac": float(np.mean(own_best)),
                    "own_margin_nats": q(own_margin),
                    "own_margin_per_token": q(np.array(own_margin) / (m * L)),
                    "own_rank_mean": float(np.mean(own_rank)), "n_own": len(own_margin)})
        if own_z:
            res["own_margin_over_dropout_sd"] = q([z for z in own_z if np.isfinite(z)])
    return res


def gap_analysis(model, groups, groups_idx, ref, eos, end_id, device, n_rep=8, with_ar=True,
                 seed=0):
    """评审第 3–4 轮的间隔分析（单个检查点）：
    - 两种代价（teacherless / AR）各自的最优-次优间隔：原始 nats、按 m·L token 归一、
      相对 dropout 复本噪声的 z、精确并列率与"噪声内并列"率、充分稳健条件 Δ>2mε；
    - 本臂自身配对（ref，名义臂为恒等、setc 为 π*）相对其余排列的间隔——同口径基线：
      固定配对训练的模型对自身配对同样会有大间隔，与 setc 比较应看这一项；
    - TL 与 AR 的决策一致率、AR 遗憾、交互项（双中心化）相关；
    - TL 交互项按答案位置的平方和占比（匹配究竟"看"的是哪些位置）。"""
    G, m = len(groups), len(groups[0])
    L = len(groups[0][0][1])
    own = own_perms_from_pairing(groups_idx, ref)
    model.eval()
    T_tl = tl_cost_tensor(model, groups, eos, end_id, device)
    C_tl = T_tl.sum(-1)
    C_ar = ar_costs(model, groups, eos, device) if with_ar else None
    reps_tl, reps_ar = [], []
    if n_rep > 0:
        torch.manual_seed(seed)
        model.train()   # 只为激活 dropout 量噪声尺度；全程 no_grad
        for _ in range(n_rep):
            reps_tl.append(tl_cost_tensor(model, groups, eos, end_id, device).sum(-1))
            if with_ar:
                reps_ar.append(ar_costs(model, groups, eos, device))
        model.eval()
    res = {"G": G, "m": m, "L": L, "n_rep": n_rep,
           "tl": gap_stats(C_tl, reps_tl, own, L)}
    if with_ar:
        res["ar"] = gap_stats(C_ar, reps_ar, own, L)
        same, dist, regret, rho, irho = [], [], [], [], []
        for g in range(G):
            p_tl, p_ar = assign(C_tl[g]), assign(C_ar[g])
            same.append(p_tl == p_ar)
            dist.append(cayley(p_tl, p_ar))
            regret.append(C_ar[g][range(m), p_tl].sum() - C_ar[g][range(m), p_ar].sum())
            a, b = C_tl[g].ravel(), C_ar[g].ravel()
            ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
            rho.append(float(np.corrcoef(ra, rb)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan)
            ia, ib = double_center(C_tl[g]).ravel(), double_center(C_ar[g]).ravel()
            irho.append(float(np.corrcoef(ia, ib)[0, 1]) if ia.std() > 0 and ib.std() > 0 else np.nan)
        res["tl_vs_ar"] = {
            "exact_agree_frac": float(np.mean(same)),
            "chance_agree": 1.0 / float(np.prod(range(1, m + 1))),
            "mean_cayley": float(np.mean(dist)),
            "chance_cayley": float(m - sum(1.0 / k for k in range(1, m + 1))),
            "ar_regret_nats_mean": float(np.mean(regret)),
            "ar_regret_nats_median": float(np.median(regret)),
            "spearman_raw_mean": float(np.nanmean(rho)),
            "pearson_interaction_mean": float(np.nanmean(irho)),
            "max_abs_raw_diff": float(np.max(np.abs(C_tl - C_ar))),
            "max_abs_interaction_diff": float(max(np.max(np.abs(double_center(C_tl[g]) -
                                                                double_center(C_ar[g])))
                                                  for g in range(G)))}
    ss = np.zeros(L)
    for g in range(G):
        ssg = np.array([(double_center(T_tl[g, :, :, t]) ** 2).sum() for t in range(L)])
        if ssg.sum() > 0:
            ss += ssg / ssg.sum()
    share = ss / max(G, 1)
    res["tl_interaction_share_by_position"] = [round(float(x), 4) for x in share]
    res["tl_interaction_top_positions"] = [int(t) for t in np.argsort(-share)[:5]]
    return res


def assignment_agreement(model, groups, eos, end_id, device):
    """teacherless 与 AR 代价各自的 Hungarian 配对的一致性及 AR 意义下的遗憾（gap 模式的轻量子集）。"""
    G, m = len(groups), len(groups[0])
    C_tl = tl_cost_tensor(model, groups, eos, end_id, device).sum(-1)
    C_ar = ar_costs(model, groups, eos, device)
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


# ---------- 两个运行的逐位比较（冻结-锁定等价检验） ----------

def compare_runs(run_a, run_b, churn_a=None, churn_b=None):
    from safetensors.torch import load_file
    res = {}
    fa = os.path.join(run_a, "final_model", "model.safetensors")
    fb = os.path.join(run_b, "final_model", "model.safetensors")
    if os.path.exists(fa) and os.path.exists(fb):
        a, b = load_file(fa), load_file(fb)
        diffs = {k: float((a[k].float() - b[k].float()).abs().max()) for k in a}
        res["weights_identical"] = all(v == 0.0 for v in diffs.values())
        res["weights_max_abs_diff"] = max(diffs.values())
    if churn_a and churn_b:
        A, B = {}, {}
        for path, D in ((churn_a, A), (churn_b, B)):
            with open_text(path) as f:
                for line in f:
                    r = json.loads(line)
                    D[(r["step"], r["group"])] = r["perm"]
        common = sorted(set(A) & set(B))
        dis = [k for k in common if A[k] != B[k]]
        res["churn_common_visits"] = len(common)
        res["churn_perm_agree_frac"] = 1 - len(dis) / max(len(common), 1)
        res["churn_first_disagree_step"] = dis[0][0] if dis else None

    def rows(run):
        out = {}
        p = os.path.join(run, "scores.jsonl")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                out[r["step"]] = r
        return out
    sa, sb = rows(run_a), rows(run_b)
    res["scores"] = [{"step": s, **{f"{k}_a": sa[s].get(k) for k in ("creativity", "test_validity", "memo_own")},
                      **{f"{k}_b": sb[s].get(k) for k in ("creativity", "test_validity", "memo_own")}}
                     for s in sorted(set(sa) & set(sb))]
    return res


def resolve_model_dirs(spec):
    """逗号分隔或 glob；按目录名中的 stepN 排序。"""
    dirs = []
    for part in spec.split(","):
        dirs += sorted(glob.glob(part)) if any(c in part for c in "*?[") else [part]
    def key(d):
        mm = re.search(r"step(\d+)", d)
        return (int(mm.group(1)) if mm else 1 << 60, d)
    return sorted(dict.fromkeys(dirs), key=key)


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
    ap.add_argument("--mode", choices=["churn", "seeds", "readout", "mi", "agree", "gap",
                                       "export_pairing", "compare"], required=True)
    ap.add_argument("--churn_dump")
    ap.add_argument("--model_dir", help="final_model 目录；gap 模式可逗号分隔或 glob（如 run/ckpts/step*）")
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
                    help="agree/gap/export/churn: 用训练 --seed 复原训练时的分组（setc 冻结配对即定义在其上）")
    ap.add_argument("--ref_pairing", default=None,
                    help="gap: 本臂自身配对（p6-pairing JSON）；缺省用 model_dir/pairing.json，再缺省为名义")
    ap.add_argument("--base_pairing", default=None,
                    help="export_pairing: 该运行自身的 --pairing_file（复合用）")
    ap.add_argument("--n_rep", type=int, default=8, help="gap: dropout 复本数（噪声尺度）")
    ap.add_argument("--no_ar", action="store_true", help="gap: 不算 AR 代价（省时）")
    ap.add_argument("--at_step", type=int, default=None, help="export_pairing: 取该步或之前最近的配对")
    ap.add_argument("--run_a")
    ap.add_argument("--run_b")
    ap.add_argument("--churn_b")
    args = ap.parse_args()

    if args.mode == "churn":
        sk = None
        if args.data_dir:
            by_group, _ = read_churn(args.churn_dump)
            sk = churn_struct_keys(args.data_dir, args.task, args.train_seed, args.m, by_group)
        res = churn_analysis(args.churn_dump, struct_keys=sk)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(json.dumps({k: v for k, v in res.items()
                          if k not in ("traj", "identity_frac", "p_eq_final", "struct_change_traj",
                                       "argmin_agree_frac")}, indent=2))
        return

    if args.mode == "export_pairing":
        with open(os.path.join(args.data_dir, "train.json"), encoding="utf-8") as f:
            n = len(json.load(f))
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train import load_pairing, write_pairing
        base = load_pairing(args.base_pairing, n) if args.base_pairing else None
        cmap, ng = export_pairing(args.churn_dump, n, args.train_seed, args.m, args.at_step, base)
        write_pairing(args.out, cmap, source=args.churn_dump, train_seed=args.train_seed,
                      m=args.m, at_step=args.at_step, n_groups_exported=ng,
                      base_pairing=args.base_pairing)
        print(json.dumps({"out": args.out, "n_groups_exported": ng,
                          "non_identity": sum(s != t for s, t in cmap.items())}))
        return

    if args.mode == "compare":
        res = compare_runs(args.run_a, args.run_b, args.churn_dump, args.churn_b)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(json.dumps({k: v for k, v in res.items() if k != "scores"}, indent=2))
        return

    if args.mode == "gap":
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train import load_pairing
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with open(os.path.join(args.data_dir, "train.json"), encoding="utf-8") as f:
            raw = json.load(f)
        results = []
        for md in resolve_model_dirs(args.model_dir):
            tok = GPT2TokenizerFast.from_pretrained(md)
            tok.pad_token = tok.eos_token
            model = GPT2LMHeadModel.from_pretrained(md).to(device)
            exs = encode_raw(raw, tok)
            gidx = build_index_groups(len(exs), args.m,
                                      np.random.default_rng(args.train_seed))[: args.G]
            groups = [[exs[j] for j in mem] for mem in gidx]
            ref_path = args.ref_pairing or (os.path.join(md, "pairing.json")
                                            if os.path.exists(os.path.join(md, "pairing.json")) else None)
            ref = load_pairing(ref_path, len(exs)) if ref_path else list(range(len(exs)))
            r = gap_analysis(model, groups, gidx, ref, tok.eos_token_id,
                             tok.convert_tokens_to_ids("</a>"), device,
                             n_rep=args.n_rep, with_ar=not args.no_ar)
            r.update({"model_dir": md, "ref_pairing": ref_path or "nominal"})
            results.append(r)
            print(json.dumps({"model_dir": md,
                              "tl_gap_median": (r["tl"].get("gap_nats") or {}).get("median"),
                              "tl_own_is_best": r["tl"].get("own_is_best_frac"),
                              "ar_own_margin_median": ((r.get("ar") or {}).get("own_margin_nats") or {}).get("median"),
                              "tl_vs_ar_agree": (r.get("tl_vs_ar") or {}).get("exact_agree_frac")}),
                  flush=True)
            del model
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
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
