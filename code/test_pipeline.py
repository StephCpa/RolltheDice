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
