"""P6 评审落地分析（2026-10-10）：
A. memo 重评分：setc_v2 按 π*_final 配对重评训练配对记忆；randperm_v2 评"组内任一目标"。
   【事件 #5 更正】首版把预测整串（含本 seed 的 hash 前缀）与各组员的 target_text 整串比较，
   而组员的 target_text 带各自的 hash 前缀 → 只可能与本 seed 的名义目标相等：
   memo_pi_final 与 memo_any_group_target 被迫退化为"π*_final 不动点且被复现"的比例
   （setc_v2：35/128，恰等于 π*_final 在这 128 个探针上的不动点数）。现改为只比较答案部分。
   更系统的版本见 train.py/evaluator.py 的 memo_own 与 rescore.py --pairing_file。
B. 差距基线：ntp_fixed 末模型在同样组上的最优-次优差距（对照 setc_v2 的 479 nats）。
   注意：ntp_fixed 从未训练 TL 目标，其 teacherless 代价是分布外读数；同口径基线应比较
   AR（teacher-forcing）代价下"本臂自身配对 vs 其余 23 个排列"的间隔，并按 token 数与
   dropout 噪声尺度归一（diagnose.py --mode gap）。
C. C01b：同一模型上 teacherless 代价与 NTP(teacher-forcing) 代价的选排列一致率与 regret。
"""
import json, sys, itertools
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, 'code')
from train import encode_raw, build_groups, mtp_seq, ntp_seq, pad_batch, MAX_LEN
from evaluator import answer_of
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from scipy.optimize import linear_sum_assignment

DATA = 'data/circle.10.9.10.10000'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = 'runs'

tok0 = GPT2TokenizerFast.from_pretrained('gpt2'); tok0.pad_token = tok0.eos_token
tok0.add_tokens(json.load(open(f'{DATA}/vocab.json', encoding='utf-8')))
raw = json.load(open(f'{DATA}/train.json', encoding='utf-8'))
exs = encode_raw(raw, tok0)
groups = build_groups(exs, 4, np.random.default_rng(42))   # 与 train.py 同调用序
text2idx = {s['input_text']: k for k, s in enumerate(raw)}
tok2idx = {tuple(p): k for k, (p, a) in enumerate(exs)}
tok2grp = {tuple(p): (g, i) for g, grp in enumerate(groups)
           for i, (p, a) in enumerate(grp)}

by_group = {}
for l in open(f'{ROOT}/server-circle-40k/setc_m4_v2/churn.jsonl', encoding='utf-8'):
    r = json.loads(l)
    by_group.setdefault(r['group'], []).append((r['step'], r['perm']))
PF = {g: sorted(v)[-1][1] for g, v in by_group.items()}


def load(model_dir):
    tok = GPT2TokenizerFast.from_pretrained(model_dir)
    tok.pad_token = tok.eos_token
    return tok, GPT2LMHeadModel.from_pretrained(model_dir).to(dev).eval()


@torch.no_grad()
def greedy_pred(tok, m, prompt):
    enc = tok(prompt, return_tensors='pt').to(dev)
    end_id = tok.convert_tokens_to_ids('</a>')
    o = m.generate(**enc, max_length=MAX_LEN, do_sample=False,
                   pad_token_id=tok.eos_token_id, eos_token_id=end_id)
    return prompt + tok.decode(o[0][enc['input_ids'].shape[1]:], skip_special_tokens=False)


@torch.no_grad()
def cost_matrices(model, grp_batch, eos):
    """返回 (C_TL, C_NTP)：G 组 m×m，teacherless 槽位代价 与 teacher-forcing 答案 NLL。"""
    G, mm = len(grp_batch), 4
    tl_seqs = [mtp_seq(p, a, eos) for grp in grp_batch for (p, a) in grp]
    ntp_seqs = [ntp_seq(pi, aj) for grp in grp_batch
                for i, (pi, _) in enumerate(grp) for j, (_, aj) in enumerate(grp)]
    C_TL = np.zeros((G, mm, mm)); C_NTP = np.zeros((G * mm * mm,))
    # NTP 代价：每 (z_i, y_j) 一条 teacher-forcing 前向
    for b0 in range(0, len(ntp_seqs), 64):
        batch = ntp_seqs[b0:b0 + 64]
        inp, lab = pad_batch(batch, eos)
        inp = inp.to(dev)
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == 'cuda')):
            lp = F.log_softmax(model(inp).logits.float(), -1)
        for b, (ids, labs) in enumerate(batch):
            pl = next(t for t, x in enumerate(labs) if x != -100)
            tgt = torch.tensor([x for x in labs if x != -100], device=dev)
            L = len(tgt)
            ll = lp[b, pl - 1:pl - 1 + L, :][torch.arange(L, device=dev), tgt]
            C_NTP[b0 + b] = -ll.sum().item()
    # TL 代价：每 seed 一次前向，槽位分布打分全部 y_j（同分布复用）
    for b0 in range(0, len(tl_seqs), 64):
        batch = tl_seqs[b0:b0 + 64]
        inp, _ = pad_batch(batch, eos)
        inp = inp.to(dev)
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == 'cuda')):
            lp = F.log_softmax(model(inp).logits.float(), -1)
        for b, (ids, labs) in enumerate(batch):
            g, i = divmod(b0 + b, mm)
            pl = next(t for t, x in enumerate(labs) if x != -100)
            L = sum(1 for x in labs if x != -100)
            rows = lp[b, pl - 1:pl - 1 + L, :]
            for j, (_, aj) in enumerate(grp_batch[g]):
                tgt = torch.tensor(aj, device=dev)
                n = min(L, len(aj))
                C_TL[g, i, j] = -rows[torch.arange(n, device=dev), tgt[-n:]].sum().item()
    return C_TL, C_NTP.reshape(G, mm, mm)


def perm_costs(C):
    return [sum(C[i][p[i]] for i in range(4)) for p in itertools.permutations(range(4))]


def main():
    res = {}

    # ---- A. memo 重评分 ----
    probes = [p for p in json.load(open(f'{DATA}/test.json', encoding='utf-8'))
              if p.get('type') == 'train'][:128]
    for arm, use_pf in (('server-circle-40k/setc_m4_v2', True),
                        ('server-circle-40k/ntp_randperm_v2', False)):
        tok, m = load(f'{ROOT}/{arm}/final_model')
        c_nom = c_pf = c_any = 0
        for p in probes:
            pred = answer_of(greedy_pred(tok, m, p['input_text']))
            idx = text2idx[p['input_text']]
            g, i = tok2grp[tuple(exs[idx][0])]
            # 只比较答案部分：target_text 带各组员自己的 hash 前缀（事件 #5）
            golds = [answer_of(raw[tok2idx[tuple(pp)]]['target_text'])
                     for (pp, _) in groups[g]]
            c_nom += int(pred == golds[i])
            c_any += int(pred in golds)
            if use_pf:
                c_pf += int(pred == golds[PF[g][i]])
        res[arm] = {'memo_nominal': c_nom / len(probes),
                    'memo_pi_final': (c_pf / len(probes)) if use_pf else None,
                    'memo_any_group_target': c_any / len(probes)}
        print(arm, res[arm])

    # ---- B+C. 差距基线 与 C01b（各模型、同样 512 组）----
    sel = groups[:512]
    for arm in ('20261009-circle-40k/ntp_fixed', 'server-circle-40k/setc_m4_v2'):
        tok, m = load(f'{ROOT}/{arm}/final_model')
        eos = tok.eos_token_id
        gaps, agree, regrets = [], [], []
        for c0 in range(0, len(sel), 64):
            gb = sel[c0:c0 + 64]
            C_TL, C_NTP = cost_matrices(m, gb, eos)
            for g in range(len(gb)):
                pc_tl = sorted(perm_costs(C_TL[g]))
                gaps.append(pc_tl[1] - pc_tl[0])
                _, p_tl = linear_sum_assignment(C_TL[g])
                _, p_ntp = linear_sum_assignment(C_NTP[g])
                agree.append(int(list(p_tl) == list(p_ntp)))
                best_ntp = min(perm_costs(C_NTP[g]))
                chose = sum(C_NTP[g][i][p_tl[i]] for i in range(4))
                regrets.append(chose - best_ntp)
        res[f'{arm}_cost'] = {
            'tl_gap_median': float(np.median(gaps)),
            'P_piTL_eq_piNTP': float(np.mean(agree)),
            'ntp_regret_median': float(np.median(regrets)),
            'ntp_regret_mean': float(np.mean(regrets))}
        print(arm, res[f'{arm}_cost'])

    json.dump(res, open(f'{ROOT}/review_analyses_20261010.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
