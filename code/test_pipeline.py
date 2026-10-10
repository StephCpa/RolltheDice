"""P6 回归测试：训练/评价/诊断语义（CPU、离线，无需下载 gpt2）。

用法：python code/test_pipeline.py   （或 pytest code/test_pipeline.py）
依赖：torch、transformers、scipy、numpy（与训练环境相同）。
用随机初始化的微型 GPT-2 与离线 WordLevel 分词器，只检查语义，不检查数值性能：
- 组级步骤的 NTP 批 = (seed_i 前缀, 被配对答案 y_π(i))（事件 #3 回归）；
- setc 向量化代价矩阵 = 逐对暴力计算；teacherless 输入不泄漏被配对答案；
- Hungarian 并列打破；评价/读取对 </a> 之后内容的截断（事件 #4 回归）；
- I(z;[y]) 插件代理的置换零分布、基于似然的 MI 估计的精确零点。
"""
import json
import os
import random
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnose  # noqa: E402
import evaluator  # noqa: E402
import train  # noqa: E402

NODES = [f"<a_{i}>" for i in range(10)]
SPECIAL = ["<mask>", "<sep>", "<a>", "</a>", "<q>", "</q>"]


def make_tok():
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    chars = list("abcdefghijklmnopqrstuvwxyz0123456789")
    vocab = {t: i for i, t in enumerate(["<|endoftext|>"] + chars + NODES + SPECIAL)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<|endoftext|>"))
    tk.pre_tokenizer = pre_tokenizers.Split(
        Regex(r"<\|endoftext\|>|<[^<>]+>|."), behavior="isolated")
    tk.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>",
                                   pad_token="<|endoftext|>")


def make_model(vocab_size, seed=0):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=vocab_size, n_positions=128, n_embd=32, n_layer=2,
                     n_head=2, eos_token_id=0, bos_token_id=0)
    return GPT2LMHeadModel(cfg)


def circle(nodes, rnd=None):
    edges = [nodes[i] + nodes[(i + 1) % len(nodes)] for i in range(len(nodes))]
    if rnd is not None:
        rnd.shuffle(edges)  # 与 datagen 一致：边序随机（canonical 结构由边序决定）
    return "<sep>".join(edges)


def synth_groups(G=3, m=4, P=3, L=5, end=1):
    # 前缀编码 (g, i)、答案编码 (g, j)，便于核对配对
    return [[([20 + g, 30 + i, 40][:P], [50 + g] + [2 + i] * (L - 2) + [end])
             for i in range(m)] for g in range(G)]


def no_scaler():
    return torch.amp.GradScaler("cuda", enabled=False)


# ---------------- 评价器 ----------------

def test_evaluator_selftest():
    assert evaluator.selftest() == 0


def test_truncate_answer():
    assert train.truncate_answer("x</a><|endoftext|><|endoftext|>") == "x</a>"
    assert train.truncate_answer("x</a>y</a>") == "x</a>"
    assert train.truncate_answer("no end") == "no end"


# ---------------- 组级训练步骤 ----------------

def test_group_step_pairing_semantics():
    """事件 #3 回归：四种组级模式的 NTP 批都必须是 (p_i, a_π(i))。"""
    model = make_model(64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    groups = synth_groups()
    captured = []
    orig = train.pad_batch
    train.pad_batch = lambda seqs, eos: (captured.append(seqs), orig(seqs, eos))[1]
    try:
        for mode in ["randperm", "setc", "tl_fixed", "tl_randperm"]:
            captured.clear()
            log = []
            _, perms = train.group_step(model, opt, no_scaler(), groups, [0, 1, 2], mode,
                                        "cpu", 0, log)
            ntp = captured[-1]
            for g in range(3):
                assert sorted(perms[g]) == list(range(4))
                for i in range(4):
                    p, a = groups[g][i][0], groups[g][perms[g][i]][1]
                    assert ntp[g * 4 + i] == (p + a, [-100] * len(p) + a), (mode, g, i)
            if mode == "tl_fixed":
                assert perms == [[0, 1, 2, 3]] * 3
            if mode in train.TL_MODES:
                assert [r["perm"] for r in log] == perms
                assert all(("argmin" in r) == (mode != "setc") for r in log)
    finally:
        train.pad_batch = orig


def test_setc_cost_matrix_matches_bruteforce():
    import scipy.optimize as so
    model = make_model(64).eval()  # 关闭 dropout，使两次前向可比
    groups = synth_groups()
    with torch.no_grad():
        brute = []
        for grp in groups:
            C = torch.zeros(4, 4)
            for i, (p, a) in enumerate(grp):
                ids, _ = train.mtp_seq(p, a, 0)
                lp = F.log_softmax(model(torch.tensor([ids])).logits[0].float(), -1)
                for j, (_, b) in enumerate(grp):
                    C[i, j] = -sum(lp[len(p) - 1 + t, b[t]] for t in range(len(b)))
            brute.append(C)
    seen = []
    orig = so.linear_sum_assignment
    so.linear_sum_assignment = lambda c: (seen.append(torch.tensor(c)), orig(c))[1]
    try:
        train.group_step(model, torch.optim.SGD(model.parameters(), lr=0.0), no_scaler(),
                         groups, [0, 1, 2], "setc", "cpu", 0, None)
    finally:
        so.linear_sum_assignment = orig
    for g in range(3):
        assert torch.allclose(seen[g].float(), brute[g], atol=1e-4)


def test_teacherless_input_does_not_leak_answer():
    model = make_model(64).eval()
    p, a = synth_groups()[0][0]
    a2 = [63] * (len(a) - 1) + a[-1:]
    with torch.no_grad():
        l1 = model(torch.tensor([train.mtp_seq(p, a, 0)[0]])).logits[0, len(p) - 1:len(p) - 1 + len(a)]
        l2 = model(torch.tensor([train.mtp_seq(p, a2, 0)[0]])).logits[0, len(p) - 1:len(p) - 1 + len(a)]
    assert torch.allclose(l1, l2)


def test_assign_tiebreak():
    C = np.full((4, 4), 3.0)
    assert train.assign(C) == [0, 1, 2, 3]          # 完全并列 → 确定性恒等置换
    rng = np.random.default_rng(0)
    outs = [tuple(train.assign(C, rng)) for _ in range(200)]
    assert all(sorted(o) == [0, 1, 2, 3] for o in outs)
    assert len(set(outs)) > 10                      # 随机打破后覆盖多种置换
    D = np.random.default_rng(1).normal(size=(4, 4))
    assert train.assign(D, rng) == train.assign(D)   # 无并列时结果不变


# ---------------- 评价流水线 ----------------

def test_run_eval_trailing_pad_counts_as_seen():
    """事件 #4 回归：批内提前停止的行带尾部 pad，已见解仍须判为 seen。"""
    tok = make_tok()
    model = make_model(len(tok))
    ans = circle(NODES[:9])
    raw = [{"input_text": "abc<q>", "target_text": "abc<q>" + ans + "</a>"}]
    probes = [{"type": "test", "input_text": "xyz<q>", "target_text": "xyz<q></a>"},
              {"type": "test", "input_text": "uvw<q>", "target_text": "uvw<q></a>"}]
    end = tok.convert_tokens_to_ids("</a>")
    seen_ids = tok(ans, add_special_tokens=False)["input_ids"] + [end]
    long_ids = tok(ans + "<sep>" + NODES[9] + NODES[0], add_special_tokens=False)["input_ids"]

    def fake_generate(input_ids, **kw):
        rows = [seen_ids + [0] * (len(long_ids) + 1 - len(seen_ids)), long_ids + [end]]
        return torch.cat([input_ids, torch.tensor(rows)], dim=1)

    model.generate = fake_generate
    with tempfile.TemporaryDirectory() as d:
        train.run_eval(model, tok, probes, raw, "circle", d, 0, "cpu", 1, 1.0, 0)
        sc = json.loads(open(os.path.join(d, "scores.jsonl")).readline())
        preds = [json.loads(x) for x in open(os.path.join(d, "preds_step0.jsonl"))]
    assert sc["test_seen"] == 0.5 and sc["test_unseen"] == 0.0, sc
    assert preds[0]["model_output"].endswith("</a>")


def test_teacherless_readout_single_end_token():
    tok = make_tok()
    ans = circle(NODES[:9]) + "</a>"
    ans_ids = tok(ans, add_special_tokens=False)["input_ids"]
    probes = [{"type": "test", "input_text": "xyz<q>", "target_text": "xyz<q></a>"}]
    P = len(tok("xyz<q>", add_special_tokens=False)["input_ids"])
    V = len(tok)

    class Oracle(torch.nn.Module):
        def forward(self, t):
            logits = torch.zeros(t.shape[0], t.shape[1], V)
            for k, tid in enumerate(ans_ids):
                logits[:, P - 1 + k, tid] = 10.0
            return type("O", (), {"logits": logits})()

    items = diagnose.teacherless_readout(Oracle(), tok, probes, "circle", len(ans_ids), 0, "cpu")
    raw = [{"input_text": "abc<q>", "target_text": "abc<q>" + circle(NODES[1:10]) + "</a>"}]
    assert evaluator.Evaluator("circle", raw).validity(items[0]["model_output"]) == 1


# ---------------- 诊断 ----------------

def test_key_of_requires_validity_and_truncates():
    raw = [{"input_text": "abc<q>", "target_text": "abc<q>" + circle(NODES[:9]) + "</a>"}]
    ev = evaluator.Evaluator("circle", raw)
    good = circle(NODES[:9]) + "</a><|endoftext|>"
    bad = circle(NODES[:4]) + "<sep>" + circle(NODES[4:9]) + "</a>"   # 两个分离环
    assert diagnose.key_of("circle", good, ev) is not None
    assert diagnose.key_of("circle", bad, ev) is None


def test_plugin_I_proxy_is_explained_by_null():
    """z 与 y 独立、样本几乎两两不同：插件 I 代理≈log2 K，但与置换零分布无差。"""
    raw = [{"input_text": "abc<q>", "target_text": "abc<q>" + circle(NODES[:9]) + "</a>"}]
    ev = evaluator.Evaluator("circle", raw)
    rnd = random.Random(0)
    samples = {}
    for h in range(32):
        lst = []
        for _ in range(16):
            lst.append(circle(rnd.sample(NODES, 9), rnd) + "</a>")
        samples[f"h{h}"] = lst
    _, _, I, _ = diagnose.entropy_probe(samples, "circle", ev)
    null = diagnose.permutation_null(samples, "circle", ev, n_perm=5)
    assert I > 4.5                                       # 看似"种子携带大量信息"
    assert abs(I - null["I_null_mean_bits"]) < 0.05      # 实为样本量封顶


def test_mi_estimate_zero_for_identical_prompts():
    tok = make_tok()
    model = make_model(len(tok)).eval()
    p = tok("abc<q>", add_special_tokens=False)["input_ids"]
    torch.manual_seed(0)
    res = diagnose.mi_estimate(model, tok, [p] * 4, 4, 1.0, 0, "cpu")
    if res["n_samples"]:  # 随机模型可能采不到 </a>；有样本时必须精确为 0
        assert abs(res["I_bits"]) < 1e-4, res


def test_seq_logprobs_matches_model_loss():
    model = make_model(64).eval()
    p, a = [5, 6, 7], [8, 9, 10, 1]
    lp = diagnose.seq_logprobs(model, [(p, a)], "cpu", 0)[0]
    with torch.no_grad():
        out = model(torch.tensor([p + a]), labels=torch.tensor([[-100] * 3 + a]))
    assert abs(lp + out.loss.item() * len(a)) < 1e-4


def test_assignment_agreement_runs():
    model = make_model(64).eval()
    res = diagnose.assignment_agreement(model, synth_groups(), 0, 1, "cpu")
    assert 0.0 <= res["exact_agree_frac"] <= 1.0 and res["mean_ar_regret_nats"] >= -1e-6


def test_churn_analysis_variants():
    recs = [{"step": s, "group": 0, "perm": [0, 1, 2, 3], "cost": 1.0, "argmin": [0, 1, 2, 3]}
            for s in (0, 10)]
    recs += [{"step": 5, "group": 0, "perm": [1, 0, 2, 3], "cost": 1.0, "members": [7, 8, 9, 10]}]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    try:
        res = diagnose.churn_analysis(f.name)
    finally:
        os.unlink(f.name)
    assert res["n_groups"] == 2 and res["mean_cayley"] == 0.0
    assert res["argmin_agree_frac"][0]["frac"] == 1.0


# ---------------- 第二轮：目标分解 / 锁定 / 回放 / 续训 / 报告 ----------------

def _step(model, groups, cfg, **kw):
    """lr=0 的一步（权重不变，便于在同一模型上比较不同目标的损失）。"""
    return train.group_step(model, torch.optim.SGD(model.parameters(), lr=0.0), no_scaler(),
                            groups, list(range(len(groups))), cfg, "cpu", 0, kw.pop("log", None),
                            **kw)


def _capture_ntp():
    captured = []
    orig_ntp, orig_w = train._ntp_loss, train._weighted_ntp_loss

    def spy(model, seqs, eos, device):
        captured.append(("hf", seqs, None))
        return orig_ntp(model, seqs, eos, device)

    def spy_w(model, seqs, weights, eos, device):
        captured.append(("w", seqs, weights))
        return orig_w(model, seqs, weights, eos, device)
    train._ntp_loss, train._weighted_ntp_loss = spy, spy_w
    return captured, (orig_ntp, orig_w)


def test_ntp_lambda_targets_and_weighting():
    model = make_model(64).eval()
    groups = synth_groups()
    cap, orig = _capture_ntp()
    try:
        res = {}
        for lam in (0.0, 1.0, 0.5):
            cap.clear()
            st, perms = _step(model, groups, train.GroupCfg("hungarian", "matched", lam, 0.0))
            res[lam] = (st, perms, cap[-1])
    finally:
        train._ntp_loss, train._weighted_ntp_loss = orig
    perms = res[0.0][1]
    assert res[1.0][1] == perms and res[0.5][1] == perms     # 同一模型 → 同一 Hungarian 结果
    nominal = [(groups[g][i][0] + groups[g][i][1], [-100] * 3 + groups[g][i][1])
               for g in range(3) for i in range(4)]
    matched = [(groups[g][i][0] + groups[g][perms[g][i]][1], [-100] * 3 + groups[g][perms[g][i]][1])
               for g in range(3) for i in range(4)]
    assert res[0.0][2][1] == matched and res[1.0][2][1] == nominal
    kind, seqs, w = res[0.5][2]
    assert kind == "w" and seqs == matched + nominal and w == [0.5] * 12 + [0.5] * 12
    # 答案等长 → 加权损失 = 两种纯目标损失的平均
    assert abs(res[0.5][0]["ntp"] - 0.5 * (res[0.0][0]["ntp"] + res[1.0][0]["ntp"])) < 1e-5


def test_lambda1_equals_incident3_objective():
    """setc --ntp_lambda 1 与事件 #3 偏差版的 NTP 目标相同：π 为组内双射时
    {(p_j, a_j): j=π(i)} 恰是该组全部名义对（只是顺序不同）。"""
    model = make_model(64).eval()
    groups = synth_groups()
    st, perms = _step(model, groups, train.GroupCfg("hungarian", "matched", 1.0, 3.0))
    buggy = [train.ntp_seq(groups[g][perms[g][i]][0], groups[g][perms[g][i]][1])
             for g in range(3) for i in range(4)]
    with torch.no_grad():
        old = float(train._ntp_loss(model, buggy, 0, "cpu"))
    assert abs(st["ntp"] - old) < 1e-5


def test_tl_target_nominal_uses_diagonal():
    model = make_model(64).eval()
    groups = synth_groups()
    st_rev, perms = _step(model, groups, train.GroupCfg("hungarian", "nominal", 0.0, 3.0))
    st_fix, _ = _step(model, groups, "tl_fixed")
    assert abs(st_rev["mtp"] - st_fix["mtp"]) < 1e-6          # TL 分支都用名义（对角）目标
    st_setc, _ = _step(model, groups, "setc")
    if any(p != [0, 1, 2, 3] for p in perms):
        assert abs(st_setc["mtp"] - st_rev["mtp"]) > 1e-7      # 匹配 TL 与名义 TL 不同


def test_lock_uses_cached_perms_and_logs_argmin():
    model = make_model(64).eval()
    groups = synth_groups()
    cached = {0: [3, 2, 1, 0], 1: [1, 0, 3, 2], 2: [0, 1, 2, 3]}
    log = []
    _, perms = _step(model, groups, train.GroupCfg("hungarian", "matched", 0.0, 3.0, lock_at=0),
                     log=log, last_perm=dict(cached))
    assert perms == [cached[0], cached[1], cached[2]]
    assert all(r["locked"] and "argmin" in r for r in log)
    _, free = _step(model, groups, "setc")
    assert [r["argmin"] for r in log] == free                # 影子 argmin = 在线 Hungarian 的选择


def test_hungarian_without_tl_loss():
    model = make_model(64).eval()
    st, perms = _step(model, synth_groups(), train.GroupCfg("hungarian", "matched", 0.0, 0.0))
    assert st["mtp"] is None and abs(st["loss"] - st["ntp"]) < 1e-9
    assert all(sorted(p) == [0, 1, 2, 3] for p in perms)


# ---- 端到端（train.main 在进程内运行；微型 GPT-2 + 字节级分词器 + 64 例 circle 数据） ----

def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs, k = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + k)
            k += 1
    return dict(zip(bs, map(chr, cs)))


_E2E = {}


def e2e_env():
    """一次性构建：微型本地 gpt2 目录与 64 例 circle 数据（datagen 同一生成逻辑）。"""
    if _E2E:
        return _E2E
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
    import datagen
    root = tempfile.mkdtemp(prefix="p6_e2e_")
    md = os.path.join(root, "tiny-gpt2")
    os.makedirs(md)
    vocab = {ch: i for i, ch in enumerate(_bytes_to_unicode().values())}
    vocab["<|endoftext|>"] = len(vocab)
    json.dump(vocab, open(os.path.join(md, "vocab.json"), "w"))
    open(os.path.join(md, "merges.txt"), "w").write("#version: 0.2\n")
    tok = GPT2TokenizerFast(os.path.join(md, "vocab.json"), os.path.join(md, "merges.txt"),
                            eos_token="<|endoftext|>", bos_token="<|endoftext|>",
                            unk_token="<|endoftext|>")
    tok.save_pretrained(md)
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=128, n_embd=32, n_layer=2,
                               n_head=2, eos_token_id=tok.eos_token_id,
                               bos_token_id=tok.eos_token_id)).save_pretrained(md)
    random.seed(1)
    np.random.seed(1)
    ev, tr, te, _, _ = datagen.circle_build(10, 9, 10, max_train_num=64, n_test=16)
    dd = os.path.join(root, "data")
    os.makedirs(dd)
    json.dump(tr, open(os.path.join(dd, "train.json"), "w"))
    json.dump(te, open(os.path.join(dd, "valid.json"), "w"))
    probes = [dict(p, type="train") for p in tr[:8]] + [dict(p, type="test") for p in te[:8]]
    json.dump(probes, open(os.path.join(dd, "test.json"), "w"))
    json.dump(ev + datagen.SPECIAL_TOKENS, open(os.path.join(dd, "vocab.json"), "w"))
    _E2E.update(root=root, model=md, data=dd)
    return _E2E


def _main(out, *extra, steps=12):
    env = e2e_env()
    train.main(["--model", env["model"], "--data_dir", env["data"], "--out", out,
                "--steps", str(steps), "--eval_every", "6", "--n_probes", "16",
                "--groups_per_step", "4", "--warmup", "2", "--log_every", "4",
                "--save_final", *extra])


def _weights(run):
    from safetensors.torch import load_file
    return load_file(os.path.join(run, "final_model", "model.safetensors"))


def test_resume_from_state_is_exact():
    env = e2e_env()
    a = os.path.join(env["root"], "straight")
    b = os.path.join(env["root"], "resumed")
    _main(a, "--arm", "setc_m4", "--state_at", "6", "--random_tiebreak")
    _main(b, "--arm", "setc_m4", "--random_tiebreak", "--resume_from", os.path.join(a, "state_step6"))
    wa, wb = _weights(a), _weights(b)
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    rows = [json.loads(x) for x in open(os.path.join(b, "scores.jsonl"))]
    assert rows[-1]["step"] == 12 and "memo_own" in rows[-1] and "nll_own" in rows[-1]
    assert os.path.exists(os.path.join(a, "train_log.jsonl"))


def test_pairing_replay_roundtrip():
    env = e2e_env()
    src = os.path.join(env["root"], "src_setc")
    _main(src, "--arm", "setc_m4", steps=8)
    fp = os.path.join(src, "final_pairing.json")
    rep = os.path.join(env["root"], "replay")
    _main(rep, "--arm", "tl_fixed_m4", "--pairing_file", fp, steps=4)
    a = json.load(open(fp))["pairs"]
    b = json.load(open(os.path.join(rep, "final_pairing.json")))["pairs"]
    assert a == b and json.load(open(os.path.join(rep, "run_meta.json")))["pairing_sha256"]


def test_export_pairing_from_churn():
    n, m = 16, 4
    groups = train.build_index_groups(n, m, np.random.default_rng(42))
    recs = [{"step": 0, "group": 0, "perm": [0, 1, 2, 3]},
            {"step": 5, "group": 0, "perm": [1, 0, 3, 2]},
            {"step": 9, "group": 2, "perm": [3, 2, 1, 0]}]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    try:
        cmap, ng = diagnose.export_pairing(f.name, n, 42, m)
        cmap4, _ = diagnose.export_pairing(f.name, n, 42, m, at_step=4)
    finally:
        os.unlink(f.name)
    g0, g2 = groups[0], groups[2]
    assert ng == 2 and cmap[g0[0]] == g0[1] and cmap[g0[2]] == g0[3] and cmap[g2[0]] == g2[3]
    assert cmap4[g0[0]] == g0[0]                              # at_step=4 取第 0 步的配对
    assert sorted(cmap.values()) == list(range(n))


def test_churn_commitment_and_final_stats():
    recs = []
    for s, p in [(0, [1, 0, 2, 3]), (10, [0, 1, 2, 3]), (20, [0, 1, 2, 3]), (30, [0, 1, 2, 3])]:
        recs.append({"step": s, "group": 0, "perm": p})
    for s, p in [(5, [1, 0, 3, 2]), (15, [1, 0, 3, 2])]:
        recs.append({"step": s, "group": 1, "perm": p})
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    try:
        res = diagnose.churn_analysis(f.name, window=1000, commit_bin=10)
    finally:
        os.unlink(f.name)
    assert res["commitment"]["hist"] == {0: 1, 10: 1}         # 组 1 在第 5 步承诺，组 0 在第 10 步
    fps = res["final_perm_stats"]
    assert fps["identity_share"] == 0.5 and fps["cycle_types"] == {"1+1+1+1": 0.5, "2+2": 0.5}
    fpw = {w["window_step"]: w["frac"] for w in res["fixed_point_frac"]}
    assert abs(fpw[0] - (2 + 4 + 4 + 4 + 0 + 0) / 24) < 1e-9          # 6 次访问的不动点均值


def test_gap_stats_own_margin_and_noise():
    rng = np.random.default_rng(0)
    G, m, L = 5, 4, 3
    C = rng.normal(size=(G, m, m)) * 0.01
    own = [[1, 0, 3, 2]] * G
    for g in range(G):
        for i in range(m):
            C[g, i, own[g][i]] -= 5.0                          # 自身配对每元低 5 nats
    reps = np.stack([C + rng.normal(size=C.shape) * 0.01 for _ in range(4)])
    st = diagnose.gap_stats(C, reps, own, L)
    assert st["own_is_best_frac"] == 1.0 and st["own_margin_nats"]["median"] > 9.0
    assert st["robust_frac_2m_eps"] == 1.0 and st["noise_tie_frac"] == 0.0
    assert abs(st["own_margin_per_token"]["median"] - st["own_margin_nats"]["median"] / (m * L)) < 1e-9


def test_report_selection_and_verdict():
    import report
    root = tempfile.mkdtemp(prefix="p6_rep_")

    def mk(name, label, seed, crea, ckpt_steps, held=None):
        d = os.path.join(root, name)
        os.makedirs(d)
        json.dump({"arm": "x", "label": label, "seed": seed, "steps": 30, "n_probes": 256},
                  open(os.path.join(d, "run_meta.json"), "w"))
        with open(os.path.join(d, "scores.jsonl"), "w") as f:
            for s, c in zip((10, 20, 30), crea):
                f.write(json.dumps({"step": s, "creativity": c, "test_validity": 0.5}) + "\n")
        for s in ckpt_steps:
            os.makedirs(os.path.join(d, "ckpts", f"step{s}"))
        if held is not None:
            os.makedirs(os.path.join(d, "heldout"))
            with open(os.path.join(d, "heldout", "scores.jsonl"), "w") as f:
                f.write(json.dumps({"step": held[0], "creativity": held[1]}) + "\n")
    for sd in (1, 2, 3):
        mk(f"ref{sd}", "ref", sd, (0.1, 0.2, 0.2), (10, 20, 30), held=(20, 0.20))
        mk(f"arm{sd}", "arm", sd, (0.3, 0.3, 0.1), (20, 30), held=(20, 0.25 + 0.01 * sd))
    sel = os.path.join(root, "sel.tsv")
    report.main(["--runs", os.path.join(root, "*"), "--ref", "ref", "--margin", "0.03",
                 "--json", os.path.join(root, "r.json"), "--emit_selection", sel])
    r = json.load(open(os.path.join(root, "r.json")))
    by = {x["run"].rsplit("/", 1)[1]: x for x in r["runs"]}
    assert by["ref1"]["selected"]["step"] == 20                 # 平局取最早
    assert by["arm1"]["selected"]["step"] == 20                 # 第 10 步无检查点，不可选
    assert r["verdicts"]["arm"]["verdict"] == "PASS" and len(open(sel).readlines()) == 6


def test_make_heldout_disjoint():
    import make_heldout
    env = e2e_env()
    out = os.path.join(env["root"], "heldout.json")
    make_heldout.main(["--data_dir", env["data"], "--n", "50", "--out", out])
    used = set()
    for fn in ("train.json", "valid.json", "test.json"):
        used |= {s["input_text"] for s in json.load(open(os.path.join(env["data"], fn)))}
    held = json.load(open(out))
    assert len(held) == 50 and not (used & {h["input_text"] for h in held})
    assert len({h["input_text"] for h in held}) == 50


def test_c01_reuse_equals_recompute():
    import throughput
    model = make_model(64).eval()
    groups = synth_groups()
    C1, n1, _, _ = throughput.costs_reuse(model, groups, 1, "cpu", 64, False)
    C2, n2, _, _ = throughput.costs_recompute(model, groups, 1, "cpu", 64, False)
    assert (n1, n2) == (12, 48) and np.max(np.abs(C1 - C2)) < 1e-4
    with torch.no_grad():
        for g in range(3):
            assert np.max(np.abs(C1[g] - diagnose.tl_cost_tensor(model, [groups[g]], 0, 1, "cpu")[0].sum(-1))) < 1e-4


# ---------------- 冲突分解：fixed 配对规则（无匹配的持久双射） ----------------

def test_fixed_rule_uses_given_perms():
    model = make_model(64).eval()
    groups = synth_groups()
    fixed = {0: [1, 0, 3, 2], 1: [2, 3, 0, 1], 2: [0, 1, 2, 3]}
    log = []
    cap, orig = _capture_ntp()
    try:
        st, perms = _step(model, groups, train.GroupCfg("fixed", "matched", 1.0, 3.0),
                          log=log, fixed_perms=fixed)
    finally:
        train._ntp_loss, train._weighted_ntp_loss = orig
    assert perms == [fixed[0], fixed[1], fixed[2]]
    assert all("argmin" in r and "locked" not in r for r in log)      # 影子 argmin 照记
    nominal = [train.ntp_seq(groups[g][i][0], groups[g][i][1]) for g in range(3) for i in range(4)]
    assert cap[-1][1] == nominal                                       # λ=1：NTP 训名义
    st_tl, _ = _step(model, groups, train.GroupCfg("fixed", "matched", 0.0, 3.0), fixed_perms=fixed)
    st_nom, _ = _step(model, groups, "tl_fixed")
    assert abs(st["mtp"] - st_tl["mtp"]) < 1e-6 and abs(st["mtp"] - st_nom["mtp"]) > 1e-7


def test_fixed_random_vs_file_and_resume():
    env = e2e_env()
    a = os.path.join(env["root"], "fixed_rand")
    _main(a, "--arm", "setc_m4", "--pair_rule", "fixed", "--fixed_pairing", "random",
          "--ntp_lambda", "1", "--state_at", "6", "--churn_dump", os.path.join(a, "churn.jsonl"))
    fp = json.load(open(os.path.join(a, "fixed_pairing.json")))["pairs"]
    assert json.load(open(os.path.join(a, "final_pairing.json")))["pairs"] == fp
    by = {}
    for line in open(os.path.join(a, "churn.jsonl")):
        r = json.loads(line)
        by.setdefault(r["group"], set()).add(tuple(r["perm"]))
    assert all(len(v) == 1 for v in by.values())                       # 每组配对恒定
    b = os.path.join(env["root"], "fixed_file")
    _main(b, "--arm", "setc_m4", "--pair_rule", "fixed", "--ntp_lambda", "1",
          "--fixed_pairing", os.path.join(a, "fixed_pairing.json"))
    c = os.path.join(env["root"], "fixed_resumed")
    _main(c, "--arm", "setc_m4", "--resume_from", os.path.join(a, "state_step6"))
    wa, wb, wc = _weights(a), _weights(b), _weights(c)
    assert all(torch.equal(wa[k], wb[k]) for k in wa)                  # 文件回放 = 随机生成（逐位）
    assert all(torch.equal(wa[k], wc[k]) for k in wa)                  # 续训逐位一致


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_")):
        try:
            fn()
            print("PASS", name)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print("FAIL", name, type(e).__name__, e)
    print("ALL OK" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
