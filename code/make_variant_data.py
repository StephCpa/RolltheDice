"""数据变体：目标多重性实验的混杂对照（plan/icml2027-plan.md §4.1b）。

dup_hash（"每答案 k 个 hash、每 hash 单目标"）：为每个训练目标再配 k-1 个新 hash。
与 ntp_2tgt（每 hash 两目标、每答案两 hash）在 (hash, 答案) 对总数上匹配，但每个 hash 仍只决定一个答案——
用来区分"每个 seed 有两个目标"与"每个答案有两个 seed / 训练对数翻倍"。
新 hash 与原 train/valid/test/heldout 的全部 hash 不相交；原训练样本保持原顺序与编号（新条目追加在后），
test.json / valid.json / vocab.json / heldout.json 原样复制——开发集与留出集与基础数据完全相同。

用法：python code/make_variant_data.py --base_dir data/circle.10.9.10.10000 --mode dup_hash --k 2 \
        --out data/circle.10.9.10.10000.dup2

graph（监督二部图因子设计，plan/icml2027-review-20261017.md §3）：给定 hash 数 N_h、每 hash 目标数 d_H、
每答案 hash 数 d_Y，构造 d_H-正则 × d_Y-正则的随机二部图（无重边），答案数 N_a = N_h·d_H/d_Y。
用来把"答案多入口"与"hash 数/训练行数翻倍"分开：例如 N_h=10000、d_Y=2 只用 5000 个答案（hash 数不变、
每答案两入口）；N_h=20000、d_H=d_Y=1 需要 20000 个答案（hash 数翻倍、仍一对一）。
- hash：N_h ≤ 基础 hash 数时取基础训练的前 N_h 个；否则追加与 train/valid/test/heldout 不相交的新 hash；
- 答案：按基础训练顺序取前 N_a 个不同答案串（图的节点按答案串定义——模型训练的是 token 序列；Circle 的
  10k 个训练答案串只有约 8.9k 个不同 canonical 结构，variant.json 同时记录结构级计数）；不够时从
  --answer_pool（另一种子生成的同任务 train.json）补充新答案串；
- (d_H, d_Y) = (1, 1) 且 N_h = N_a ≤ 基础数时保留基础配对（与基础数据的前 N_h 行逐行相同）；
- test.json 中 type=train 的探针改写为该 hash 在新图中的（第一个）目标，或在 hash 不再训练时删除；
  test/valid/heldout 的测试 hash 不变。d_H > 1 时同一 hash 有多行，训练器自动不计 memo_own/nll。

  python code/make_variant_data.py --base_dir data/circle.10.9.10.10000 --mode graph \
      --n_hashes 10000 --hash_degree 1 --answer_degree 2 --out data/circle.10.9.10.10000.g_h10k_d1x2
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _answer(s):
    return s["target_text"].split("<q>", 1)[1]


def _canon_fn(task):
    if task in ("circle", "line"):
        from evaluator import circle_canonicalize
        return lambda a: circle_canonicalize(a) or a
    return lambda a: a


def _structure_degrees(rows, task):
    """结构级（canonical）统计：不同结构数，以及每个结构平均对应的训练 hash 数。"""
    canon = _canon_fn(task)
    hashes_per = {}
    for r in rows:
        hashes_per.setdefault(canon(_answer(r)), set()).add(r["input_text"])
    n = len(hashes_per)
    return {"n_structures": n, "mean_hashes_per_structure": round(sum(map(len, hashes_per.values())) / n, 4)}


def _used_hashes(args):
    paths = [os.path.join(args.base_dir, fn) for fn in ("train.json", "valid.json", "test.json", "heldout.json")]
    used = set()
    for p in paths + list(args.exclude):
        if os.path.exists(p):
            used |= {s["input_text"].split("<q>")[0] for s in json.load(open(p, encoding="utf-8"))}
        else:
            assert p in paths, f"--exclude 文件不存在：{p}"
    return used


def _new_hash(rnd, used, hash_len, chars=string.ascii_lowercase + string.digits):
    while True:
        h = "".join(rnd.choice(chars) for _ in range(hash_len))
        if h not in used:
            used.add(h)
            return h


def biregular_edges(n_h, n_a, d_h, d_y, rnd, max_rounds=200):
    """d_h-正则（hash 侧）× d_y-正则（答案侧）的随机二部图，无重边；返回每个 hash 的答案下标列表。"""
    assert n_h * d_h == n_a * d_y, "需满足 N_h·d_H = N_a·d_Y"
    assert d_h <= n_a, "d_H 不能超过答案数"
    stubs = [y for y in range(n_a) for _ in range(d_y)]
    rnd.shuffle(stubs)
    adj = [stubs[i * d_h:(i + 1) * d_h] for i in range(n_h)]
    for _ in range(max_rounds):          # 局部修复：把重复的桩与随机另一 hash 的桩交换
        bad = [i for i in range(n_h) if len(set(adj[i])) < d_h]
        if not bad:
            return adj
        for i in bad:
            seen = set()
            for p, y in enumerate(adj[i]):
                if y not in seen:
                    seen.add(y)
                    continue
                for _ in range(1000):
                    j = rnd.randrange(n_h)
                    q = rnd.randrange(d_h)
                    z = adj[j][q]
                    if j != i and z not in adj[i] and y not in adj[j]:
                        adj[i][p], adj[j][q] = z, y
                        seen.add(z)
                        break
    raise RuntimeError("biregular_edges: 修复未收敛（d_H/d_Y 相对答案数过大？）")


def build_graph_variant(args, train, used, hash_len):
    base_hashes = [s["input_text"].split("<q>")[0] for s in train]
    n_h, d_h, d_y = args.n_hashes or len(train), args.hash_degree, args.answer_degree
    assert (n_h * d_h) % d_y == 0, "N_h·d_H 必须能被 d_Y 整除"
    n_a = n_h * d_h // d_y
    answers = list(dict.fromkeys(_answer(s) for s in train))
    if len(answers) < n_a:
        assert args.answer_pool, f"需要 {n_a} 个答案，基础数据只有 {len(answers)} 个：请给 --answer_pool"
        keys = set(answers)
        for s in json.load(open(args.answer_pool, encoding="utf-8")):
            a = _answer(s)
            if a not in keys:
                keys.add(a)
                answers.append(a)
            if len(answers) >= n_a:
                break
        assert len(answers) >= n_a, f"--answer_pool 去重后仍不足 {n_a} 个答案"
    answers = answers[:n_a]
    rnd = random.Random(args.seed)
    hashes = base_hashes[:n_h] + [_new_hash(rnd, used, hash_len) for _ in range(max(0, n_h - len(train)))]
    keep_base = (d_h, d_y) == (1, 1) and n_h <= len(train) and \
        all(_answer(train[i]) == answers[i] for i in range(n_h))
    adj = [[i] for i in range(n_h)] if keep_base else biregular_edges(n_h, n_a, d_h, d_y, rnd)
    rows = [{"input_text": h + "<q>", "target_text": h + "<q>" + answers[y]}
            for h, ys in zip(hashes, adj) for y in ys]
    first = {r["input_text"]: r["target_text"] for r in reversed(rows)}
    meta = {"n_hashes": n_h, "n_answers": n_a, "hash_degree": d_h, "answer_degree": d_y,
            "n_edges": len(rows), "n_new_hashes": max(0, n_h - len(train)),
            "kept_base_pairing": keep_base, "answer_pool": args.answer_pool, "task": args.task,
            **_structure_degrees(rows, args.task)}
    return rows, first, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["dup_hash", "graph"], default="dup_hash")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--task", choices=["circle", "line", "sibling"], default="circle",
                    help="graph：variant.json 中结构级计数用的 canonical 规则")
    ap.add_argument("--n_hashes", type=int, default=None, help="graph：训练 hash 数（缺省=基础训练行数）")
    ap.add_argument("--hash_degree", type=int, default=1, help="graph：每个 hash 的目标数 d_H")
    ap.add_argument("--answer_degree", type=int, default=1, help="graph：每个答案的 hash 数 d_Y")
    ap.add_argument("--answer_pool", default=None, help="graph：答案不够时补充用的同任务 train.json")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="新 hash 还须避开的探针文件（如另名的 fresh heldout）；base_dir 下的四个文件总是避开")
    args = ap.parse_args(argv)
    if args.mode == "graph":
        return main_graph(args)
    assert args.k >= 2

    train = json.load(open(os.path.join(args.base_dir, "train.json"), encoding="utf-8"))
    used = _used_hashes(args)
    hash_len = max(len(s["input_text"].split("<q>")[0]) for s in train)
    assert hash_len > 0, "dup_hash 需要 hash 条件数据"
    chars = string.ascii_lowercase + string.digits
    rnd = random.Random(args.seed)
    extra = []
    for s in train:
        answer = s["target_text"].split("<q>", 1)[1]
        for _ in range(args.k - 1):
            while True:
                h = "".join(rnd.choice(chars) for _ in range(hash_len))
                if h not in used:
                    used.add(h)
                    break
            extra.append({"input_text": h + "<q>", "target_text": h + "<q>" + answer})
    os.makedirs(args.out, exist_ok=True)
    out_train = train + extra
    with open(os.path.join(args.out, "train.json"), "w", encoding="utf-8") as f:
        json.dump(out_train, f)
    for fn in ("test.json", "valid.json", "vocab.json", "heldout.json"):
        p = os.path.join(args.base_dir, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(args.out, fn))
    sha = hashlib.sha256(open(os.path.join(args.out, "train.json"), "rb").read()).hexdigest()
    meta = {"mode": args.mode, "k": args.k, "base_dir": args.base_dir, "seed": args.seed,
            "n_train_base": len(train), "n_train_out": len(out_train), "train_sha256": sha}
    with open(os.path.join(args.out, "variant.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))


def main_graph(args):
    train = json.load(open(os.path.join(args.base_dir, "train.json"), encoding="utf-8"))
    used = _used_hashes(args)
    hash_len = max(len(s["input_text"].split("<q>")[0]) for s in train)
    assert hash_len > 0, "graph 需要 hash 条件数据"
    rows, first, gmeta = build_graph_variant(args, train, used, hash_len)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f)
    probes = json.load(open(os.path.join(args.base_dir, "test.json"), encoding="utf-8"))
    out_probes = []
    for p in probes:
        if p.get("type") == "train":
            if p["input_text"] not in first:
                continue
            p = {**p, "target_text": first[p["input_text"]]}
        out_probes.append(p)
    with open(os.path.join(args.out, "test.json"), "w", encoding="utf-8") as f:
        json.dump(out_probes, f)
    for fn in ("valid.json", "vocab.json", "heldout.json"):
        p = os.path.join(args.base_dir, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(args.out, fn))
    sha = hashlib.sha256(open(os.path.join(args.out, "train.json"), "rb").read()).hexdigest()
    meta = {"mode": "graph", "base_dir": args.base_dir, "seed": args.seed, **gmeta,
            "n_train_base": len(train), "n_train_out": len(rows),
            "n_train_probes_out": sum(p.get("type") == "train" for p in out_probes),
            "train_sha256": sha}
    with open(os.path.join(args.out, "variant.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
