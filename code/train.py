"""P6 WP1/WP2: 干净重实现的训练器。

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
  tl_hybrid    原 teacherless 混合（1 NTP+3 MTP）     16 例/步（忠实原文展开）
  组级臂（G=--groups_per_step 组 × m 例/步，组成员固定；各臂只是 GroupCfg 预设）：
  ntp_randperm       random    / —       / NTP 匹配目标 / TL 权重 0     G1 对照
  ntp_fixed_grouped  nominal   / —       / NTP 名义     / TL 权重 0     组批组织版 ntp_fixed（流水线自检）
  setc_m4            hungarian / matched / NTP 匹配目标 / TL 权重 3     集合耦合（修正版 v2）
  tl_fixed_m4        nominal   / nominal / NTP 名义     / TL 权重 3     setc 的最小对照（N-TL / N-NTP）
  tl_randperm_m4     random    / matched / NTP 匹配目标 / TL 权重 3
  任一组级臂可用 --pair_rule / --tl_target / --ntp_lambda / --mtp_weight 覆盖预设，例如：
  setc_m4 --ntp_lambda 1      = 匹配 TL + 名义 NTP：与事件 #3 偏差版 setc（本地 v1）目标精确相同
                                （π 是组内双射 → 组内 {(p_j, a_j) : j=π(i)} 就是全部名义对）；
  setc_m4 --tl_target nominal = 名义 TL + 匹配 NTP（反向组合）；
  setc_m4 --ntp_lambda 0.5    = NTP 分支按 0.5/0.5 混合匹配与名义目标（总监督权重不变）。
  setc_m4 --pair_rule fixed --fixed_pairing random --ntp_lambda 1
                              = TL 训每组一个固定随机双射、NTP 训名义：有分支冲突、无匹配；
  ntp_fixed_grouped --pair_rule fixed --fixed_pairing random --ntp_lambda 0.5
                              = 仅 NTP 的双目标（名义 + 固定随机伙伴），无 TL。
  组大小由 --m 决定（臂名中的 m4 只是标签）：--m 64 --groups_per_step 1 即整批匹配。

干预/回放（评审第 2–4 轮的冻结-锁定-回放设计）：
  --state_at S1,S2  在这些步保存完整训练状态（模型+优化器+调度器+全部随机流+分组/顺序/配对）；
  --resume_from DIR 从完整状态续训（仅组级臂），可同时改变 --lock_at / --ntp_lambda 等作分支；
  --lock_at S       从第 S 步起 hungarian 配对改用各组上次的配对（Hungarian 仍计算并记为 argmin，
                    TL 前向本就为 TL 损失所需，故锁定不改变任何随机数消耗 → 冻结时与在线分支逐位一致）；
  --pairing_file F  用 F 中的 seed→target 配对替换数据集名义配对（π* 回放 / 迁移）；
                    组级臂分组只依赖样本编号与 --seed，同 seed 下 π*（组内置换）可直接回放。
  --ckpt_every K    每 K 步保存仅模型检查点（检查点选择 → 留出集评测、按阶段诊断）。

评价额外记录（scores.jsonl）：creativity（论文主指标）、memo_own（对本臂实际训练配对的复现，
仅比较答案部分）、train_seen、nll_own / nll_nominal（固定 256 个训练 seed 上的逐 token NLL）。
偏差登记：vendored simpletransformers 需 2021 版 transformers（无 cp312 轮子），
按上述语义干净重实现；环境与差异记入 run_meta.json。
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace

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


def build_index_groups(n, m, rng):
    idx = list(range(n))
    rng.shuffle(idx)
    return [idx[i:i + m] for i in range(0, len(idx) - m + 1, m)]


def build_groups(exs, m, rng):
    return [[exs[j] for j in mem] for mem in build_index_groups(len(exs), m, rng)]


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


# ---------------- 配对文件（π* 导出 / 回放） ----------------

def load_pairing(path, n):
    """读取 {"pairs": [[seed, target], ...]}（训练样本编号）。返回 R：R[s] = s 的训练目标编号。
    未出现的 seed 保持名义配对。"""
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    pairs = obj["pairs"] if isinstance(obj, dict) else obj
    R = list(range(n))
    for s, t in pairs:
        s, t = int(s), int(t)
        assert 0 <= s < n and 0 <= t < n, f"配对越界: {s}->{t}"
        R[s] = t
    return R


def apply_pairing(exs, R):
    return [(exs[s][0], exs[R[s]][1]) for s in range(len(exs))]


def write_pairing(path, cmap, **meta):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"format": "p6-pairing-v1", **meta,
                   "pairs": [[int(s), int(t)] for s, t in sorted(cmap.items())]}, f)


def fixed_overlap(raw, R, groups_idx, fixed_perms, task):
    """修订 №1 §2 要求登记：固定伙伴 R 与名义 N 的条目级重合率（应为 0）与结构级重合率
    （伙伴目标与名义目标 canonical 结构相同的 seed 占比）。"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluator import circle_canonicalize

    def key(t):
        a = raw[t]["target_text"].split("<q>")[1].split("</a>")[0]
        return circle_canonicalize(a) if task == "circle" else a
    n_e = n_s = n = 0
    for g, mem in enumerate(groups_idx):
        for i, e in enumerate(mem):
            t_nom, t_fix = R[e], R[mem[fixed_perms[g][i]]]
            n += 1
            n_e += int(t_fix == t_nom)
            n_s += int(t_fix == t_nom or key(t_fix) == key(t_nom))
    return {"entry_overlap_with_nominal": n_e / max(n, 1),
            "structure_overlap_with_nominal": n_s / max(n, 1)}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ---------------- 组级步骤 ----------------

@dataclass
class GroupCfg:
    """组级臂的训练目标。
    pair_rule : nominal（名义）| random（每次访问随机双射）| hungarian（teacherless 代价上的最优双射）
                | fixed（每组一个持久双射，来自 --fixed_pairing：random 或配对文件；不做匹配）
                | multi（仅 NTP 的多目标：每个 seed 训 k 个不同目标 {名义, σ(i), σ²(i), …}，σ 为组的固定伙伴置换）
    tl_target : matched（TL 损失用 pair_rule 的配对）| nominal（TL 损失用名义配对）
    ntp_lambda: NTP 分支中名义目标的权重；0 = 全用匹配目标，1 = 全用名义目标，中间值按权混合
                （逐 token 加权平均，总监督权重不变）
    mtp_weight: TL 损失权重（0 = 不训练 TL；hungarian 仍会做无梯度前向以取代价）
    lock_at   : ≥0 时，从该步起 hungarian 改用各组上次的配对（影子 argmin 照记）
    n_targets : multi 的目标数 k（含名义）
    target_mode: multi 的组织方式：weighted（每次访问 k 个目标各 1/k 权重，k×G×m 条序列）
                | sample（每次访问每个 seed 随机取其中一个目标，G×m 条序列——与 ntp_fixed 同算力）"""
    pair_rule: str = "nominal"
    tl_target: str = "matched"
    ntp_lambda: float = 0.0
    mtp_weight: float = 3.0
    lock_at: int = -1
    n_targets: int = 2
    target_mode: str = "weighted"


PRESETS = {
    "randperm": GroupCfg("random", "matched", 0.0, 0.0),
    "setc": GroupCfg("hungarian", "matched", 0.0, 3.0),
    "tl_fixed": GroupCfg("nominal", "nominal", 0.0, 3.0),
    "tl_randperm": GroupCfg("random", "matched", 0.0, 3.0),
    "ntp_fixed_grouped": GroupCfg("nominal", "nominal", 0.0, 0.0),
}
ARM_PRESET = {"ntp_randperm": "randperm", "setc_m4": "setc", "tl_fixed_m4": "tl_fixed",
              "tl_randperm_m4": "tl_randperm", "ntp_fixed_grouped": "ntp_fixed_grouped"}
TL_MODES = ("setc", "tl_fixed", "tl_randperm")  # 兼容旧名：带 TL 损失的预设


def assign(C, tiebreak_rng=None):
    """Hungarian 求 argmin_π Σ_i C[i, π(i)]，返回 π（列表）。
    linear_sum_assignment 对完全并列的代价确定性地返回恒等置换（fp16 logits
    舍入或模型尚未区分 seed 时常见）；tiebreak_rng 给定时先随机置换列再求解，
    使并列被均匀打破。"""
    from scipy.optimize import linear_sum_assignment
    if tiebreak_rng is None:
        r, c = linear_sum_assignment(C)
        assert list(r) == list(range(C.shape[0]))
        return list(map(int, c))
    sh = tiebreak_rng.permutation(C.shape[1])
    _, c = linear_sum_assignment(C[:, sh])
    return [int(sh[k]) for k in c]


def perm_power(p, r):
    """σ^r：对 seed i 返回 σ(σ(…σ(i)))（r 次）。"""
    q = list(range(len(p)))
    for _ in range(r):
        q = [p[x] for x in q]
    return q


def _ntp_loss(model, seqs, eos, device):
    inp, lab = pad_batch(seqs, eos)
    inp, lab = inp.to(device), lab.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
        out = model(inp, labels=lab)
    return out.loss.float()


def _weighted_ntp_loss(model, seqs, weights, eos, device):
    """逐 token 交叉熵按序列权重加权平均：Σ w_b·Σ_t ℓ_bt / Σ w_b·n_b。
    组内答案等长时等于 Σ_k w_k·mean_k（各目标集的平均 token 损失按权混合）。"""
    inp, lab = pad_batch(seqs, eos)
    inp, lab = inp.to(device), lab.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
        logits = model(inp).logits
    logits = logits[:, :-1].float()
    tgt = lab[:, 1:]
    tok = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                          ignore_index=-100, reduction="none").view(tgt.shape)
    w = torch.tensor(weights, device=device, dtype=torch.float32).unsqueeze(1) * (tgt != -100)
    return (tok * w).sum() / w.sum()


def group_step(model, opt, scaler, groups_ex, group_ids, cfg, device, step,
               churn_log, mtp_weight=None, matrix_log=None, tiebreak_rng=None,
               members=None, last_perm=None, perm_keys=None, fixed_perms=None):
    """一步组级训练。cfg 为 GroupCfg 或预设名（"randperm"/"setc"/"tl_fixed"/"tl_randperm"/
    "ntp_fixed_grouped"）；mtp_weight 若给定则覆盖 cfg。
    group_ids: 每组编号（churn 记录用）；perm_keys: last_perm 的键（默认 group_ids；--regroup 时用成员元组）。
    last_perm: dict，记录各组上次实际使用的配对（锁定与 churn 统计用；就地更新）。
    fixed_perms: pair_rule="fixed" 时各组（按 perm_keys）的持久配对。
    返回 (stats, perms)：stats 含 loss / ntp / mtp / n_changed / n_revisit。"""
    if isinstance(cfg, str):
        cfg = PRESETS[cfg]
    if mtp_weight is not None:
        cfg = replace(cfg, mtp_weight=mtp_weight)
    eos = model.config.eos_token_id
    G = len(groups_ex)
    m = len(groups_ex[0])
    ident = list(range(m))
    keys = list(perm_keys) if perm_keys is not None else [int(g) for g in group_ids]
    locked = (cfg.pair_rule == "hungarian" and cfg.lock_at >= 0 and step >= cfg.lock_at
              and last_perm is not None)

    perms = []
    if cfg.pair_rule == "random":
        for _ in range(G):
            pi = list(range(m))
            random.shuffle(pi)
            perms.append(pi)
    elif cfg.pair_rule == "nominal":
        perms = [list(range(m)) for _ in range(G)]
    elif cfg.pair_rule in ("fixed", "multi"):
        perms = [list(fixed_perms[k]) for k in keys]
    elif cfg.pair_rule != "hungarian":
        raise ValueError(cfg.pair_rule)

    train_tl = cfg.mtp_weight > 0
    mtp_loss = None
    if train_tl or cfg.pair_rule == "hungarian":
        # ---- teacherless 前向（每 seed 一次；m² 代价是同次前向的 gather，向量化）----
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
        with torch.set_grad_enabled(train_tl):
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(device == "cuda")):
                logits = model(inp).logits
            B = G * m
            # 每个答案槽 t 的预测来自位置 P-1+t（因果 LM 内在 shift）
            p_t = torch.tensor(p_lens, device=device).unsqueeze(1) - 1 + \
                torch.arange(L, device=device)
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
        # Hungarian 在 detach 代价上选择（非 hungarian 规则或锁定时只作影子记录 argmin）
        C_np = C.detach().cpu().numpy()
        dump_matrices = (matrix_log is not None
                         and step % matrix_log["every"] == 0)
        for g in range(G):
            hung = assign(C_np[g], tiebreak_rng)
            if cfg.pair_rule == "hungarian":
                cached = last_perm.get(keys[g]) if locked else None
                perms.append(list(cached) if cached is not None else hung)
            pi = perms[g]
            if churn_log is not None:
                rec = {"step": step, "group": int(group_ids[g]), "perm": pi,
                       "cost": float(C_np[g][np.arange(m), pi].sum())}
                if cfg.pair_rule != "hungarian" or locked:
                    rec["argmin"] = hung
                if locked:
                    rec["locked"] = True
                if members is not None:
                    rec["members"] = members[g]
                churn_log.append(rec)
            if dump_matrices:
                mrec = {"step": step, "group": int(group_ids[g]),
                        "C": C_np[g].round(4).tolist(), "perm": pi}
                if members is not None:
                    mrec["members"] = members[g]
                matrix_log["records"].append(mrec)
        if train_tl:
            # 被配对条目的 TL 损失（梯度只流过被选条目；argmin 不可微，EM 式）
            tl_perms = perms if cfg.tl_target == "matched" else [ident] * G
            perm_t = torch.tensor(tl_perms, device=device)                   # [G, m]
            sel_ll = ll[torch.arange(G, device=device).view(G, 1, 1),
                        torch.arange(m, device=device).view(1, m, 1),
                        perm_t.view(G, m, 1),
                        torch.arange(L, device=device).view(1, 1, L)]        # [G, m, L]
            mtp_loss = -sel_ll.mean()

    # ---- NTP 前向（前缀取 seed i，答案取目标 y_j；名义目标 j=i）----
    def ntp_seqs(pp):
        return [ntp_seq(groups_ex[g][i][0], groups_ex[g][pp[g][i]][1])
                for g in range(G) for i in range(m)]

    lam = 0.0 if cfg.pair_rule == "nominal" else float(cfg.ntp_lambda)
    if cfg.pair_rule == "multi":
        k = cfg.n_targets
        # 目标集合顺序：σ, σ², …, σ^(k-1), 名义（k=2 时与 fixed+λ=0.5 的批顺序完全相同）
        target_sets = [[perm_power(perms[g], r) for g in range(G)] for r in range(1, k)]
        target_sets.append([ident] * G)
        if cfg.target_mode == "weighted":
            seqs, w = [], []
            for ts in target_sets:
                seqs += ntp_seqs(ts)
                w += [1.0 / k] * (G * m)
            ntp_loss = _weighted_ntp_loss(model, seqs, w, eos, device)
        else:
            pick = [[target_sets[random.randrange(k)][g][i] for i in range(m)] for g in range(G)]
            ntp_loss = _ntp_loss(model, ntp_seqs(pick), eos, device)
    elif lam == 0.0:
        ntp_loss = _ntp_loss(model, ntp_seqs(perms), eos, device)
    elif lam == 1.0:
        ntp_loss = _ntp_loss(model, ntp_seqs([ident] * G), eos, device)
    else:
        ntp_loss = _weighted_ntp_loss(model, ntp_seqs(perms) + ntp_seqs([ident] * G),
                                      [1.0 - lam] * (G * m) + [lam] * (G * m), eos, device)

    loss = ntp_loss + cfg.mtp_weight * mtp_loss if train_tl else ntp_loss

    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()

    n_changed = n_revisit = 0
    if last_perm is not None:
        for g in range(G):
            prev = last_perm.get(keys[g])
            if prev is not None:
                n_revisit += 1
                n_changed += int(list(prev) != perms[g])
            last_perm[keys[g]] = list(perms[g])
    stats = {"loss": float(loss.detach()), "ntp": float(ntp_loss.detach()),
             "mtp": float(mtp_loss.detach()) if mtp_loss is not None else None,
             "n_changed": n_changed, "n_revisit": n_revisit}
    return stats, perms


# ---------------- 评价 ----------------

def truncate_answer(s):
    """截到首个 </a>（含）。批量 generate 中提前停止的行会被补 pad_token
    （<|endoftext|>）；原实现 predict() 同样做 split("</a>")[0]+"</a>"。
    无 </a> 的输出保持原样（评价器判为无效，与原实现补 </a> 的差别已登记）。"""
    return s.split("</a>")[0] + "</a>" if "</a>" in s else s


def run_eval(model, tok, probes, raw, task, out_dir, step, device, n_samples, temp, eos,
             save_preds=True, do_sample=None, own_targets=None, extra=None):
    """do_sample=None 时沿用原逻辑（n_samples>1 才采样，否则 greedy）。
    own_targets: {训练探针 input_text: 本臂实际训练配对的 target_text}（用于 memo_own）。
    extra: 并入该行 scores 的附加字段（如 nll_own）。"""
    model.eval()
    end_id = tok.convert_tokens_to_ids("</a>")
    tok.padding_side = "left"
    texts = [p["input_text"] for p in probes for _ in range(n_samples)]
    sample = n_samples > 1 if do_sample is None else do_sample
    outs_flat = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], return_tensors="pt", padding=True).to(device)
            gkw = dict(max_length=MAX_LEN, do_sample=sample,
                       pad_token_id=eos, eos_token_id=end_id)
            if sample:
                gkw["temperature"] = temp
            o = model.generate(**enc, **gkw)
            for r in range(o.shape[0]):
                outs_flat.append(truncate_answer(
                    tok.decode(o[r][enc["input_ids"].shape[1]:], skip_special_tokens=False)))
    items = []
    for k, p in enumerate(probes):
        outs = [p["input_text"] + outs_flat[k * n_samples + r]
                for r in range(n_samples)]
        it = {"type": p.get("type", "test"),
              "model_output": outs if n_samples > 1 else outs[0],
              "target_text": p["target_text"]}
        if own_targets is not None and it["type"] == "train" and p["input_text"] in own_targets:
            it["own_target_text"] = own_targets[p["input_text"]]
        items.append(it)
    model.train()
    if save_preds:
        # 保存原始生成，评价器修订后可离线重算全部指标（无需重新生成）
        with open(os.path.join(out_dir, f"preds_step{step}.jsonl"), "w", encoding="utf-8") as f:
            for p, it in zip(probes, items):
                f.write(json.dumps({"input_text": p["input_text"], **it},
                                   ensure_ascii=False) + "\n")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluator import Evaluator
    scores = Evaluator(task, raw).eval_items(items, n_gen=n_samples)
    if extra:
        scores.update(extra)
    scores["step"] = step
    with open(os.path.join(out_dir, "scores.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(scores, ensure_ascii=False) + "\n")
    print("[eval]", step, json.dumps(scores, ensure_ascii=False), flush=True)
    return scores


def pair_nll(model, pairs, eos, device, bs=64):
    """逐 token 平均 NLL（teacher forcing，eval 模式、无梯度、不消耗随机数）。"""
    was_training = model.training
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            chunk = [ntp_seq(p, a) for p, a in pairs[i:i + bs]]
            k = sum(len(a) for _, a in pairs[i:i + bs])
            tot += float(_ntp_loss(model, chunk, eos, device)) * k
            n += k
    if was_training:
        model.train()
    return tot / max(n, 1)


# ---------------- 检查点 / 完整状态 ----------------

def save_model_dir(path, model, tok, cmap=None, **pairing_meta):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    if cmap is not None:
        write_pairing(os.path.join(path, "pairing.json"), cmap, **pairing_meta)


def save_train_state(path, **state):
    os.makedirs(path, exist_ok=True)
    torch.save(state, os.path.join(path, "train_state.pt"))


def git_commit():
    """代码版本：git HEAD；非 git 副本（如服务器同步目录）回退到环境变量 P6_CODE_SHA。"""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return os.environ.get("P6_CODE_SHA")


# ---------------- 主流程 ----------------

GROUP_ARMS = tuple(ARM_PRESET)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["ntp_noseed", "ntp_fixed", "tl_hybrid", *GROUP_ARMS])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--task", choices=["circle", "sibling", "line"], default="circle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None, help="运行标签（汇总报告按此分组，默认=arm）")
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
    # 组级目标（None = 用预设或续训状态中的值）
    ap.add_argument("--pair_rule", choices=["nominal", "random", "hungarian", "fixed", "multi"],
                    default=None)
    ap.add_argument("--n_targets", type=int, default=None, help="pair_rule=multi 的每 seed 目标数 k（含名义）")
    ap.add_argument("--target_mode", choices=["weighted", "sample"], default=None,
                    help="multi：weighted（k 目标各 1/k，k 倍序列）| sample（每次访问随机取一，同 ntp_fixed 算力）")
    ap.add_argument("--fixed_pairing", default=None,
                    help="pair_rule=fixed/multi 的配对：random（每组一个均匀随机双射，约 25%% seed 与名义重合）"
                         "| cycle（每组一个均匀随机 m-循环：σ^r 互不相同，multi 的 k≥3 需要）"
                         "| derangement（每组一个均匀随机错排：无 seed 与名义重合——修订 №1 §2 的 tl_RN 规格）"
                         "| p6-pairing JSON（seed→target，须落在同组内，如 setc 的 final_pairing.json）")
    ap.add_argument("--fixed_seed", type=int, default=None,
                    help="random/derangement 的随机流种子（缺省 seed+2；修订 №1 规定 777，与训练种子无关）")
    ap.add_argument("--tl_target", choices=["matched", "nominal"], default=None)
    ap.add_argument("--ntp_lambda", type=float, default=None,
                    help="NTP 分支名义目标权重：0=匹配目标，1=名义目标（setc_m4 --ntp_lambda 1 重建偏差版）")
    ap.add_argument("--mtp_weight", type=float, default=None,
                    help="组级臂 TL 损失权重（预设：TL 臂 3.0 = 原 1:3 混合比；NTP 臂 0）")
    ap.add_argument("--lock_at", type=int, default=None,
                    help="从该步起 hungarian 配对锁定为各组上次的配对（冻结-锁定检验）")
    ap.add_argument("--random_tiebreak", action="store_true",
                    help="Hungarian 前随机置换列，均匀打破并列（默认关闭以保持已有运行可复现）")
    ap.add_argument("--regroup", action="store_true",
                    help="组级臂每轮（遍历完全部组后）重新随机分组；churn 记录附 members")
    ap.add_argument("--pairing_file", default=None,
                    help="用该文件的 seed→target 配对替换名义配对（π* 回放）")
    # 检查点 / 续训
    ap.add_argument("--ckpt_every", type=int, default=0, help="每 K 步保存仅模型检查点（0=关）")
    ap.add_argument("--state_at", default="", help="逗号分隔的步数：保存完整可续训状态（组级臂）")
    ap.add_argument("--resume_from", default=None, help="完整状态目录（state_step*/）")
    ap.add_argument("--deterministic", action="store_true",
                    help="torch 确定性算法（CUDA 上较慢；跨进程逐位复现时用）")
    ap.add_argument("--log_every", type=int, default=50, help="train_log.jsonl 记录间隔")
    ap.add_argument("--nll_subset", type=int, default=256, help="nll_own/nll_nominal 用的训练 seed 数")
    ap.add_argument("--no_save_preds", action="store_true",
                    help="不保存每次评价的原始生成 preds_step*.jsonl")
    args = ap.parse_args(argv)
    is_group = args.arm in GROUP_ARMS
    if not is_group:
        for k in ("pair_rule", "tl_target", "ntp_lambda", "mtp_weight", "lock_at",
                  "n_targets", "target_mode"):
            if getattr(args, k) is not None:
                ap.error(f"--{k} 只适用于组级臂")
        if args.resume_from:
            ap.error("--resume_from 只支持组级臂（DataLoader 臂的批次顺序无法逐位续接）")
    if args.ntp_lambda is not None and not 0.0 <= args.ntp_lambda <= 1.0:
        ap.error("--ntp_lambda 须在 [0, 1]")
    if (args.pair_rule in ("fixed", "multi")) != bool(args.fixed_pairing) and not args.resume_from:
        ap.error("--pair_rule fixed/multi 与 --fixed_pairing 须同时给出")
    if args.pair_rule in ("fixed", "multi") and args.regroup:
        ap.error("--pair_rule fixed/multi 需要持久分组，不能与 --regroup 同用")
    if args.n_targets is not None and not 2 <= args.n_targets <= args.m:
        ap.error("--n_targets 须在 [2, m]")
    if args.pairing_file and args.arm in ("tl_hybrid", "ntp_noseed"):
        ap.error("--pairing_file 不适用于 tl_hybrid / ntp_noseed")
    args.state_at = sorted({int(s) for s in args.state_at.split(",") if s.strip()})
    return args


def resolve_group_cfg(args, saved=None):
    base = GroupCfg(**saved) if saved is not None else PRESETS[ARM_PRESET[args.arm]]
    over = {k: getattr(args, k) for k in ("pair_rule", "tl_target", "ntp_lambda",
                                           "mtp_weight", "lock_at", "n_targets", "target_mode")
            if getattr(args, k) is not None}
    return replace(base, **over)


def main(argv=None):
    args = parse_args(argv)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_group = args.arm in GROUP_ARMS

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    state = None
    if args.resume_from:
        state = torch.load(os.path.join(args.resume_from, "train_state.pt"),
                           map_location="cpu", weights_only=False)
        assert state["arm"] == args.arm, f"续训状态来自 {state['arm']}，与 --arm 不符"
        tok = GPT2TokenizerFast.from_pretrained(args.resume_from)
        tok.pad_token = tok.eos_token
        model = GPT2LMHeadModel.from_pretrained(args.resume_from)
    else:
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
    n = len(exs)
    R = load_pairing(args.pairing_file, n) if args.pairing_file else list(range(n))
    exs_used = apply_pairing(exs, R) if args.pairing_file else exs

    probes_all = json.load(open(os.path.join(args.data_dir, "test.json"), encoding="utf-8"))
    tr_p = [p for p in probes_all if p.get("type") == "train"][: args.n_probes // 2]
    te_p = [p for p in probes_all if p.get("type") == "test"][: args.n_probes // 2]
    probes = tr_p + te_p
    text2idx = {s["input_text"]: k for k, s in enumerate(raw)}
    hashes_unique = len(text2idx) == n

    cfg = resolve_group_cfg(args, state["group_cfg"] if state else None) if is_group else None
    # 本臂"自己的配对"：名义/hungarian 有定义；random 没有（每次访问都变）
    own_defined = hashes_unique and (not is_group or cfg.pair_rule != "random")
    current_map = {s: R[s] for s in range(n)} if own_defined else None
    nll_idx = sorted(np.random.default_rng(12345).choice(
        n, min(args.nll_subset, n), replace=False).tolist()) if hashes_unique else []

    meta = {"arm": args.arm, "label": args.label or args.arm, "task": args.task,
            "data_dir": args.data_dir,
            "steps": args.steps, "bs": args.bs, "lr": args.lr, "seed": args.seed,
            "m": args.m, "groups_per_step": args.groups_per_step, "model": args.model,
            "group_cfg": asdict(cfg) if cfg else None,
            "random_tiebreak": args.random_tiebreak, "regroup": args.regroup,
            "pairing_file": args.pairing_file, "fixed_pairing": args.fixed_pairing,
            "fixed_seed": args.fixed_seed,
            "pairing_sha256": sha256_of(args.pairing_file) if args.pairing_file else None,
            "resume_from": args.resume_from, "resume_step": state["step"] if state else None,
            "ckpt_every": args.ckpt_every, "state_at": args.state_at,
            "deterministic": args.deterministic, "n_probes": len(probes),
            "n_train_examples": n, "vocab_size": len(tok),
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0)
            if device == "cuda" else "cpu", "git_commit": git_commit(),
            "argv": sys.argv[1:] if argv is None else list(argv),
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(args.out, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, ensure_ascii=False), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(args.warmup, 1)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    tl_arm = is_group and (cfg.mtp_weight > 0 or cfg.pair_rule == "hungarian")
    churn_log = [] if tl_arm else None
    matrix_log = ({"every": 50, "records": []} if tl_arm else None)
    # 独立随机流：开启 --random_tiebreak 不改变分组/顺序所用的 rng 序列
    tiebreak_rng = np.random.default_rng(args.seed + 1) if args.random_tiebreak else None
    save_preds = not args.no_save_preds
    step_times = []
    print_every = max(50, args.steps // 40)
    t0_all = time.time()
    step = 0
    win = {"loss": [], "ntp": [], "mtp": [], "changed": 0, "revisit": 0}

    def pairing_meta():
        return {"arm": args.arm, "label": args.label or args.arm, "seed": args.seed,
                "step": step, "rule": cfg.pair_rule if cfg else "nominal",
                "pairing_file": args.pairing_file}

    def do_eval():
        extra = {}
        if nll_idx:
            extra["nll_nominal"] = pair_nll(model, [exs[s] for s in nll_idx], eos, device)
            if current_map is not None:
                extra["nll_own"] = pair_nll(
                    model, [(exs[s][0], exs[current_map[s]][1]) for s in nll_idx], eos, device)
        own_targets = None
        if current_map is not None:
            own_targets = {p["input_text"]: raw[current_map[text2idx[p["input_text"]]]]["target_text"]
                           for p in tr_p if p["input_text"] in text2idx}
        run_eval(model, tok, probes, raw, args.task, args.out, step, device,
                 args.n_samples_probe, args.sample_temp, eos, save_preds,
                 own_targets=own_targets, extra=extra)

    def log_window(lr):
        if not win["loss"]:
            return
        rec = {"step": step, "loss": float(np.mean([float(x) for x in win["loss"]])),
               "ntp": float(np.mean([float(x) for x in win["ntp"]])), "lr": lr,
               "sec_per_step": float(np.mean(step_times[-args.log_every:])) if step_times else None}
        if win["mtp"]:
            rec["mtp"] = float(np.mean(win["mtp"]))
        if win["revisit"]:
            rec["perm_change_frac"] = win["changed"] / win["revisit"]
        with open(os.path.join(args.out, "train_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        win.update({"loss": [], "ntp": [], "mtp": [], "changed": 0, "revisit": 0})

    def after_step(loss_val, extra_state=None):
        if step % args.log_every == 0:
            log_window(sched.get_last_lr()[0])
        if step % print_every == 0:
            print(f"step {step} loss {loss_val:.4f} "
                  f"sec/step {np.mean(step_times[-100:]) if step_times else 0:.3f}", flush=True)
        if step % args.eval_every == 0:
            do_eval()
        if args.ckpt_every and step % args.ckpt_every == 0:
            save_model_dir(os.path.join(args.out, "ckpts", f"step{step}"), model, tok,
                           current_map, **pairing_meta())
        if step in args.state_at:
            path = os.path.join(args.out, f"state_step{step}")
            save_model_dir(path, model, tok, current_map, **pairing_meta())
            if extra_state is not None:
                save_train_state(
                    path, step=step, arm=args.arm, seed=args.seed, group_cfg=asdict(cfg),
                    opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                    py_random=random.getstate(), np_random=np.random.get_state(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else None,
                    rng=rng.bit_generator.state,
                    tiebreak=tiebreak_rng.bit_generator.state if tiebreak_rng else None,
                    current_map=current_map, **extra_state)

    model.train()
    if is_group:
        last_perm = {}
        if state is not None:
            groups_idx, order, n_epochs = state["groups_idx"], state["order"], state["n_epochs"]
            last_perm = state["last_perm"]
            if state["current_map"] is not None and current_map is not None:
                current_map = state["current_map"]
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            scaler.load_state_dict(state["scaler"])
            rng.bit_generator.state = state["rng"]
            if tiebreak_rng is not None and state["tiebreak"] is not None:
                tiebreak_rng.bit_generator.state = state["tiebreak"]
            random.setstate(state["py_random"])
            np.random.set_state(state["np_random"])
            torch.set_rng_state(state["torch_rng"])
            if device == "cuda" and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            step = state["step"]
        else:
            groups_idx = build_index_groups(n, args.m, rng)
            order = []
            n_epochs = 0
        groups = [[exs_used[j] for j in mem] for mem in groups_idx]
        fixed_perms = None
        if cfg.pair_rule in ("fixed", "multi"):
            if state is not None and state.get("fixed_perms") is not None:
                fixed_perms = state["fixed_perms"]
            elif args.fixed_pairing in ("random", "derangement", "cycle"):
                # 独立随机流，不扰动分组/顺序；derangement 用拒绝抽样得到均匀随机错排；
                # cycle：随机次序 o 上的循环 o[j]→o[j+1]（均匀随机 m-循环）
                frng = np.random.default_rng(args.seed + 2 if args.fixed_seed is None
                                             else args.fixed_seed)
                fixed_perms = {}
                for g in range(len(groups_idx)):
                    p = frng.permutation(args.m).tolist()
                    if args.fixed_pairing == "cycle":
                        o, p = p, [0] * args.m
                        for j in range(args.m):
                            p[o[j]] = o[(j + 1) % args.m]
                    while args.fixed_pairing == "derangement" and any(p[i] == i for i in range(args.m)):
                        p = frng.permutation(args.m).tolist()
                    fixed_perms[g] = p
            else:
                Fp = load_pairing(args.fixed_pairing, n)
                fixed_perms = {}
                for g, mem in enumerate(groups_idx):
                    pos = {R[e]: k for k, e in enumerate(mem)}
                    missing = [e for e in mem if Fp[e] not in pos]
                    assert not missing, f"--fixed_pairing 的目标不在同组内（组 {g}）：须来自同一 --seed 的分组"
                    fixed_perms[g] = [pos[Fp[e]] for e in mem]
            if cfg.pair_rule == "multi":
                if cfg.mtp_weight > 0:
                    raise SystemExit("pair_rule=multi 只用于纯 NTP 多目标（--mtp_weight 0 / ntp_fixed_grouped）")
                for g, p_ in fixed_perms.items():
                    pw = [perm_power(p_, r) for r in range(1, cfg.n_targets)]
                    ok = all(all(q[i] != i for i in range(args.m)) for q in pw) and \
                        len({tuple(q) for q in pw}) == len(pw)
                    if not ok:
                        raise SystemExit(f"组 {g} 的伙伴置换给不出 {cfg.n_targets} 个互异目标"
                                         "（k≥3 请用 --fixed_pairing cycle）")
            write_pairing(os.path.join(args.out, "fixed_pairing.json"),
                          {mem[i]: R[mem[fixed_perms[g][i]]] for g, mem in enumerate(groups_idx)
                           for i in range(args.m)},
                          source=args.fixed_pairing, seed=args.seed, m=args.m,
                          fixed_seed=args.fixed_seed,
                          **fixed_overlap(raw, R, groups_idx, fixed_perms, args.task))
        while step < args.steps:
            if not order:
                if args.regroup and n_epochs > 0:
                    groups_idx = build_index_groups(n, args.m, rng)
                    groups = [[exs_used[j] for j in mem] for mem in groups_idx]
                order = list(range(len(groups)))
                rng.shuffle(order)
                n_epochs += 1
            pick = order[: args.groups_per_step]
            order = order[args.groups_per_step:]
            batch_groups = [groups[g] for g in pick]
            members = [groups_idx[g] for g in pick] if args.regroup else None
            keys = [tuple(groups_idx[g]) for g in pick] if args.regroup else None
            t0 = time.time()
            stats, perms = group_step(model, opt, scaler, batch_groups, pick, cfg, device,
                                      step, churn_log, matrix_log=matrix_log,
                                      tiebreak_rng=tiebreak_rng, members=members,
                                      last_perm=last_perm, perm_keys=keys,
                                      fixed_perms=fixed_perms)
            if current_map is not None:
                for gl, gid in enumerate(pick):
                    mem = groups_idx[gid]
                    for i in range(args.m):
                        current_map[mem[i]] = R[mem[perms[gl][i]]]
            sched.step()
            step += 1
            if step > 50:
                step_times.append(time.time() - t0)
            win["loss"].append(stats["loss"])
            win["ntp"].append(stats["ntp"])
            if stats["mtp"] is not None:
                win["mtp"].append(stats["mtp"])
            win["changed"] += stats["n_changed"]
            win["revisit"] += stats["n_revisit"]
            after_step(stats["loss"], extra_state=dict(
                groups_idx=groups_idx, order=list(order), n_epochs=n_epochs,
                last_perm=dict(last_perm), regroup=args.regroup, m=args.m,
                fixed_perms=fixed_perms,
                groups_per_step=args.groups_per_step) if step in args.state_at else None)
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
                for (p, a) in exs_used:
                    units.append((p, a, "ntp"))
                    for _ in range(3):
                        units.append((p, a, "mtp"))
        else:
            units = [(p, a, "ntp") for (p, a) in exs_used]
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
                lv = loss.detach()  # 不在每步同步 GPU（与原循环吞吐一致），记录时再转 float
                win["loss"].append(lv)
                win["ntp"].append(lv)
                after_step(lv)

    thr = {"sec_per_step_mean": float(np.mean(step_times)) if step_times else None,
           "sec_per_step_p50": float(np.median(step_times)) if step_times else None,
           "total_hours": (time.time() - t0_all) / 3600, "steps": args.steps,
           "start_step": state["step"] if state else 0}
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
    if current_map is not None:
        write_pairing(os.path.join(args.out, "final_pairing.json"), current_map, **pairing_meta())
    if args.save_final:
        save_model_dir(os.path.join(args.out, "final_model"), model, tok,
                       current_map, **pairing_meta())
    if args.steps % args.eval_every != 0:  # 否则循环内已在 args.steps 评价过（避免重复行）
        do_eval()


if __name__ == "__main__":
    main()
