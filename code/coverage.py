"""新颖性的机会基线：结构空间大小、训练覆盖率、复制倍率（plan/graph-factorial-review-20261020.md）。

creativity = V × (1 − seen/V) × u，其中 u = 去重比例。若有效输出相对"是否见过"像结构空间上的均匀抽样，
seen/V 的机会水平就是训练覆盖率 c = T/S（T = 训练答案中的不同结构数，S = 结构空间大小）。
- Circle（n 节点有向环、边序随机）/ Line（n 节点有向路径）：canonical 按首次出现重标定，轨道数 S = (n−1)!
  （Circle 9 节点 → 8! = 40,320；Line 10 节点 → 9! = 362,880）。只统计恰为 n 节点的结构；其他长度的有效
  输出必然"未见"，不进入机会模型。
- Sibling：S = Σ_a |b1(a)|·|b2(a)|（需显式图目录）；seen 按答案串精确匹配。
复制倍率 r = (seen/V) / c：≈1 表示有效输出相对训练集的重合度与均匀抽样无异；>1 表示偏向复现训练答案。
C@c0 = V·(1 − c0·r)·u：把各单元放到同一参照覆盖率 c0 下比较（缺省 c0 = 参照臂的覆盖率）。

用法：
  python code/coverage.py --runs "runs/line-phase1/*/s*" --data_root data --ref ntp_fixed --out cov.md --json cov.json
  python code/coverage.py --runs "runs/sib/*/s*" --sibling_graph <graph_dir> --ref ntp_fixed
"""
import argparse
import glob
import json
import math
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluator import circle_canonicalize, load_sibling_graph  # noqa: E402


def answer_of(row):
    a = row["target_text"].split("<q>", 1)[1]
    return a[:-4] if a.endswith("</a>") else a


def n_nodes(answer):
    return len(set(re.findall(r"<a_\d+>", answer)))


def structure_space(task, train_rows, graph=None):
    """返回 (S, T, n)：结构空间大小、训练中不同结构数、（Circle/Line）节点数。"""
    answers = [answer_of(r) for r in train_rows]
    if task in ("circle", "line"):
        n = st.mode(n_nodes(a) for a in answers)
        S = math.factorial(n - 1)
        T = len({circle_canonicalize(a + "</a>") for a in answers if n_nodes(a) == n})
        return S, T, n
    if task == "sibling":
        assert graph is not None, "Sibling 需要 --sibling_graph"
        b1, b2 = graph
        S = sum(len(b1[a]) * len(b2.get(a, ())) for a in b1)
        return S, len(set(answers)), None
    raise ValueError(task)


def run_label(run):
    parts = run.rstrip("/").split("/")
    return parts[-2] if re.fullmatch(r"s\d+", parts[-1]) and len(parts) > 1 else parts[-1]


def resolve_data_dir(d, data_root):
    if os.path.exists(os.path.join(d, "train.json")):
        return d
    if data_root:
        alt = os.path.join(data_root, os.path.basename(d.rstrip("/")))
        if os.path.exists(os.path.join(alt, "train.json")):
            return alt
    raise FileNotFoundError(f"找不到训练数据 {d}（可用 --data_root 指向本地副本）")


def analyse_run(run, data_root=None, graph=None, cache=None):
    meta = json.load(open(os.path.join(run, "run_meta.json"), encoding="utf-8"))
    rows = [json.loads(line) for line in open(os.path.join(run, "heldout", "scores.jsonl"), encoding="utf-8")
            if line.strip()]
    h = rows[-1]
    task = meta.get("task", "circle")
    ddir = resolve_data_dir(meta["data_dir"], data_root)
    key = (task, os.path.abspath(ddir))
    if cache is None or key not in cache:
        train = json.load(open(os.path.join(ddir, "train.json"), encoding="utf-8"))
        val = structure_space(task, train, graph)
        if cache is not None:
            cache[key] = val
    else:
        val = cache[key]
    S, T, n = val
    V, seen, C = h["test_validity"], h["test_seen"], h["creativity"]
    novel = V - seen
    c = T / S
    share = seen / V if V else float("nan")
    return {"run": run, "label": run_label(run), "seed": meta.get("seed"), "task": task, "step": h.get("step"),
            "S": S, "T": T, "n_nodes": n, "coverage": c, "creativity": C, "validity": V, "seen": seen,
            "seen_share": share, "copy_ratio": share / c if V else float("nan"),
            "uniqueness": C / novel if novel > 0 else float("nan")}


def at_coverage(r, c0):
    return r["validity"] * (1 - c0 * r["copy_ratio"]) * r["uniqueness"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="运行目录 glob（含 run_meta.json 与 heldout/scores.jsonl）")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--sibling_graph", default=None)
    ap.add_argument("--ref", default="ntp_fixed", help="参照标签：其覆盖率作为 c0")
    ap.add_argument("--c0", type=float, default=None, help="直接指定参照覆盖率（覆盖 --ref）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    graph = load_sibling_graph(args.sibling_graph) if args.sibling_graph else None   # (b1 字典, b2 字典)
    runs = sorted({r for pat in args.runs for r in glob.glob(pat)
                   if os.path.exists(os.path.join(r, "heldout", "scores.jsonl"))})
    cache = {}
    res = [analyse_run(r, args.data_root, graph, cache) for r in runs]
    if not res:
        raise SystemExit("没有带 heldout/scores.jsonl 的运行")
    ref = [r for r in res if r["label"] == args.ref]
    c0 = args.c0 if args.c0 is not None else (st.mean(r["coverage"] for r in ref) if ref else None)
    for r in res:
        r["creativity_at_c0"] = at_coverage(r, c0) if c0 is not None else None

    labels = sorted({r["label"] for r in res})
    agg = {}
    for lab in labels:
        rs = [r for r in res if r["label"] == lab]
        agg[lab] = {k: st.mean(r[k] for r in rs) for k in
                    ("coverage", "creativity", "validity", "seen_share", "copy_ratio", "uniqueness")}
        agg[lab]["creativity_at_c0"] = st.mean(r["creativity_at_c0"] for r in rs) if c0 is not None else None
        agg[lab]["n"] = len(rs)
        agg[lab]["S"] = rs[0]["S"]
    lines = [f"# 新颖性机会基线（c0 = {c0:.4f}）" if c0 is not None else "# 新颖性机会基线", "",
             "| 标签 | n | S | 覆盖率 c | creativity | validity | seen/V | 复制倍率 r | 去重 u | creativity@c0 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for lab in labels:
        a = agg[lab]
        c0s = f"{a['creativity_at_c0']:.3f}" if a["creativity_at_c0"] is not None else "—"
        lines.append(f"| {lab} | {a['n']} | {a['S']} | {a['coverage']:.4f} | {a['creativity']:.3f} | "
                     f"{a['validity']:.3f} | {a['seen_share']:.3f} | {a['copy_ratio']:.2f} | "
                     f"{a['uniqueness']:.3f} | {c0s} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text)
    if args.json:
        json.dump({"c0": c0, "aggregate": agg, "runs": res}, open(args.json, "w", encoding="utf-8"), indent=2)
    return agg


if __name__ == "__main__":
    main()
