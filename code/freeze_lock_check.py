"""P6 冻结-锁定一致性检查（评审指示的第一步，短程）。

从已冻结的 setc_m4_v2 末模型出发各跑 N 步：A 在线（每步 Hungarian）vs B 锁定
（churn.jsonl 缓存的 π_final）。同 torch 种子、同数据顺序、同优化器构造。
RNG 说明：Hungarian 不消耗 torch 随机数；两臂前向完全相同——若在线排列等于
π_final，两臂应逐步产生相同损失；任何分歧提示未记录的差异。
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, 'code')
from train import encode_raw, build_groups, mtp_seq, ntp_seq, pad_batch
from scipy.optimize import linear_sum_assignment
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

N_STEPS = 50
DEV = 'cuda'
DATA = 'data/circle.10.9.10.10000'
MODEL = 'runs/server-circle-40k/setc_m4_v2/final_model'
M, L = 4, 27

tok = GPT2TokenizerFast.from_pretrained(MODEL); tok.pad_token = tok.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL).to(DEV)
eos = tok.eos_token_id
raw = json.load(open(f'{DATA}/train.json', encoding='utf-8'))
exs = encode_raw(raw, tok)
groups = build_groups(exs, 4, np.random.default_rng(42))

by_group = {}
for l in open('runs/server-circle-40k/setc_m4_v2/churn.jsonl', encoding='utf-8'):
    r = json.loads(l)
    by_group.setdefault(r['group'], []).append((r['step'], r['perm']))
PF = {g: sorted(v)[-1][1] for g, v in by_group.items()}


def run(locked, seed=1234):
    torch.manual_seed(seed)
    r2 = np.random.default_rng(999)
    order = []
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
    model.train()
    losses, mism, tot = [], 0, 0
    for step in range(N_STEPS):
        if not order:
            order = list(range(len(groups)))
            r2.shuffle(order)
        pick = order[:16]
        order = order[16:]
        gb = [groups[g] for g in pick]
        G = len(gb)
        seqs = [mtp_seq(p, a, eos) for grp in gb for (p, a) in grp]
        inp, _ = pad_batch(seqs, eos)
        inp = inp.to(DEV)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(inp).logits
        lp = F.log_softmax(logits.float(), -1)
        p_lens = [len(p) for grp in gb for (p, a) in grp]
        p_t = torch.tensor(p_lens, device=DEV).unsqueeze(1) - 1 + torch.arange(L, device=DEV)
        b_t = torch.arange(G * M, device=DEV).unsqueeze(1).expand(-1, L)
        rows = lp[b_t, p_t]                 # [B, L, V]：各 seed 的答案槽位分布（先 gather 位置！）
        tgt = torch.tensor([[a for (_, a) in grp] for grp in gb], device=DEV)
        g_i = torch.arange(G, device=DEV).view(G, 1, 1, 1) * M + \
            torch.arange(M, device=DEV).view(1, M, 1, 1)
        t_t = torch.arange(L, device=DEV).view(1, 1, 1, L)
        tgt_b = tgt.unsqueeze(1).expand(G, M, M, L)
        ll = rows[g_i.expand(G, M, M, L), t_t.expand(G, M, M, L), tgt_b]
        C_np = (-ll.sum(-1)).detach().cpu().numpy()
        perms = []
        for g in range(G):
            if locked:
                c = list(PF[pick[g]])
            else:
                _, c = linear_sum_assignment(C_np[g])
                c = list(map(int, c))
                tot += 1
                if c != list(PF[pick[g]]):
                    mism += 1
            perms.append(c)
        perm_t = torch.tensor(perms, device=DEV)
        sel_ll = ll[torch.arange(G, device=DEV).view(G, 1, 1),
                    torch.arange(M, device=DEV).view(1, M, 1),
                    perm_t.view(G, M, 1),
                    torch.arange(L, device=DEV).view(1, 1, L)]
        mtp_loss = -sel_ll.mean()
        ntp_seqs = [ntp_seq(gb[g][i][0], gb[g][perms[g][i]][1])
                    for g in range(G) for i in range(M)]
        inp2, lab2 = pad_batch(ntp_seqs, eos)
        inp2, lab2 = inp2.to(DEV), lab2.to(DEV)
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(inp2, labels=lab2)
        loss = out.loss.float() + 3.0 * mtp_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return losses, mism, tot


la, mism, tot = run(locked=False)
lb, _, _ = run(locked=True)
la, lb = np.array(la), np.array(lb)
res = {
    'n_steps': N_STEPS,
    'online_vs_pf_mismatch_frac': mism / max(tot, 1),
    'max_abs_loss_diff': float(np.abs(la - lb).max()),
    'loss_equal_within_1e-4': bool(np.allclose(la, lb, atol=1e-4)),
    'note': 'lr=1e-5 短程；优化器状态未保存（两臂同构造，可比逐步损失）',
}
print(json.dumps(res, indent=1))
json.dump(res, open('runs/server-circle-40k/freeze_lock_check.json', 'w'), indent=1)
