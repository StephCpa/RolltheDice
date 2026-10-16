"""汇总报告：轨迹摘要 + 开发集检查点选择 → 留出集 + 跨训练种子聚合 + 预注册判定。

评审第 4–5 轮协议（plan/stage0b-plan.md §3）：
- 主指标 creativity = 去重有效新结构数 / 测试生成数（论文口径；旧 scores 无此键时由
  n_unique_valid_novel / 测试探针数推得并标注）；
- 每个运行按预注册规则在开发集（训练中每次评价的 128 个测试探针）上选检查点：主指标最大，
  平局取最早；只在有可评测检查点（ckpts/stepN 或 final_model）的步中选；
- 选中的检查点在留出集上一次性评测（rescore.py → <run>/heldout/scores.jsonl，step=所选步），
  报告以留出集数值为准；开发集峰值只作描述（取最大会高估）；
- 同时报告终点、峰值（及其步）、轨迹平均（AUC/步长），不把 40k 终点当唯一比较口径；
- 与参照臂按训练种子配对比较：n ≥ min_seeds、全部为留出集数值、均值差 ≥ margin 且每个种子差 > 0 才判 PASS。

用法：
  python code/report.py --runs "runs/stage0b/*/*" --emit_selection sel.tsv          # 第 1 步：选检查点
  （对 sel.tsv 每行跑 rescore.py --probe_file heldout.json --out <run>/heldout --step <step>）
  python code/report.py --runs "runs/stage0b/*/*" --selection sel.tsv \
      --ref ntp_fixed --margin 0.03 --out report.md
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

SUMMARY_METRICS = ["creativity", "test_validity", "test_novel_valid", "memo_own",
                   "train_seen", "train_memorization", "nll_own", "nll_nominal"]


def display_path(path):
    """Return a portable path for reports and JSON manifests."""
    return os.path.normpath(path).replace("\\", "/")


def sha256_file(path):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_selection(path):
    """Read a frozen selection manifest without requiring checkpoints locally."""
    if not path:
        return {}
    result = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise SystemExit(f"selection manifest line {lineno} must have 3 tab-separated fields")
            run, step, ckpt = fields
            if not re.fullmatch(r"[0-9]+", step):
                raise SystemExit(f"selection manifest line {lineno} has invalid step {step!r}")
            result[os.path.realpath(os.path.abspath(run))] = {
                "step": int(step), "ckpt_dir": display_path(ckpt)
            }
    return result


def load_rows(path, n_test_default):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["step"]] = r          # 同一步的重复行（旧版末尾重复评价）取最后一行
    for r in rows.values():
        if "test_novel_valid" not in r and "test_unseen" in r:
            r["test_novel_valid"] = r["test_unseen"]
        if "creativity" not in r and "n_unique_valid_novel" in r:
            r["creativity"] = r["n_unique_valid_novel"] / (r.get("n_test_gen") or n_test_default)
            r["creativity_derived"] = True
    return [rows[s] for s in sorted(rows)]


def traj_summary(rows, metric):
    pts = [(r["step"], r[metric]) for r in rows if r.get(metric) is not None]
    if not pts:
        return None
    steps = np.array([p[0] for p in pts], float)
    vals = np.array([p[1] for p in pts], float)
    k = int(np.argmax(vals))        # 首个最大值 = 最早
    auc = float(np.sum((vals[1:] + vals[:-1]) / 2 * np.diff(steps)) / (steps[-1] - steps[0])) \
        if len(steps) > 1 else float(vals[0])
    return {"endpoint": float(vals[-1]), "endpoint_step": int(steps[-1]),
            "peak": float(vals[k]), "peak_step": int(steps[k]), "auc_mean": auc}


def ckpt_steps(run):
    out = set()
    for d in glob.glob(os.path.join(run, "ckpts", "step*")):
        m = re.search(r"step(\d+)$", d)
        if m:
            out.add(int(m.group(1)))
    return out


def select_checkpoint(run, rows, metric, final_step):
    avail = ckpt_steps(run)
    has_final = os.path.isdir(os.path.join(run, "final_model"))
    cands = [r for r in rows if r.get(metric) is not None]
    usable = [r for r in cands if r["step"] in avail or (has_final and r["step"] == final_step)]
    no_ckpt = not usable
    if no_ckpt:          # 本地没有任何检查点：仍按规则选步作描述，但无法留出评测
        usable = cands
    if not usable:
        return None
    best = max(usable, key=lambda r: (r[metric], -r["step"]))   # 最大值，平局取最早
    s = best["step"]
    ck = None if no_ckpt else (os.path.join(run, "ckpts", f"step{s}") if s in avail
                               else os.path.join(run, "final_model"))
    return {"step": s, "ckpt_dir": ck, "dev": best, "no_ckpt": no_ckpt,
            "candidates": len(usable), "restricted_to_ckpts": (not no_ckpt) and len(usable) < len(cands)}


def heldout_row(run, name, step):
    p = os.path.join(run, name, "scores.jsonl")
    if not os.path.exists(p):
        return None
    hit = None
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("step") == step:
                hit = r
    return hit


def discover(specs):
    runs = []
    for spec in specs.split(","):
        for d in sorted(glob.glob(spec)):
            if os.path.exists(os.path.join(d, "run_meta.json")) and \
                    os.path.exists(os.path.join(d, "scores.jsonl")):
                runs.append(d)
    return list(dict.fromkeys(runs))


def summarize_run(run, args, frozen_selection=None):
    meta = json.load(open(os.path.join(run, "run_meta.json"), encoding="utf-8"))
    n_test = (meta.get("n_probes") or 256) // 2
    rows = load_rows(os.path.join(run, "scores.jsonl"), n_test)
    if args.label_by == "dir":
        # <标签>/s<种子> 布局取上一级目录名；否则取运行目录名。
        # glob 在 Windows 返回反斜杠路径；使用 os.path 而不是手工 split，
        # 否则 --label_by dir 会把每个运行误当成一个标签，无法配对裁决。
        norm_run = os.path.normpath(run)
        run_name = os.path.basename(norm_run)
        parent_name = os.path.basename(os.path.dirname(norm_run))
        label = parent_name if re.fullmatch(r"s\d+", run_name) else run_name
    else:
        label = meta.get("label") or meta.get("arm")
    sel = select_checkpoint(run, rows, args.primary, meta.get("steps"))
    frozen = (frozen_selection or {}).get(os.path.realpath(os.path.abspath(run)))
    if frozen:
        selected_row = next((r for r in rows if r.get("step") == frozen["step"]), None)
        if selected_row is None:
            raise SystemExit(
                f"selection manifest step {frozen['step']} has no matching score row for {run}"
            )
        sel = {"step": frozen["step"], "ckpt_dir": frozen["ckpt_dir"],
               "dev": selected_row, "no_ckpt": not os.path.isdir(frozen["ckpt_dir"]),
               "candidates": 1, "restricted_to_ckpts": True}
    out = {"run": display_path(run), "label": label, "seed": meta.get("seed"), "arm": meta.get("arm"),
           "task": meta.get("task"), "data_dir": meta.get("data_dir"),
           "training_git_commit": meta.get("git_commit"),
           "group_cfg": meta.get("group_cfg"), "steps": meta.get("steps"),
           "traj": {m: traj_summary(rows, m) for m in SUMMARY_METRICS},
           "creativity_derived": any(r.get("creativity_derived") for r in rows)}
    if sel:
        out["selected"] = {"step": sel["step"], "ckpt_dir": sel["ckpt_dir"],
                           "restricted_to_ckpts": sel["restricted_to_ckpts"],
                           "dev": {m: sel["dev"].get(m) for m in SUMMARY_METRICS}}
        if frozen:
            out["selected"]["selection_manifest"] = display_path(args.selection)
        h = heldout_row(run, args.heldout_name, sel["step"])
        out["heldout"] = {m: h.get(m) for m in SUMMARY_METRICS + ["n_test_gen"]} if h else None
        if h:
            # Keep score provenance alongside the aggregate metrics.  The
            # corrected Line rows identify the actual checkpoint and probe;
            # step alone is only a label and is insufficient for replay.
            out["heldout_identity"] = {
                k: h.get(k) for k in ("step", "model_dir", "probe_file", "n_test_gen", "n_train_gen")
            }
    return out


def mean_se(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    a = np.array(xs, float)
    se = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None
    return {"mean": float(a.mean()), "se": se, "n": len(a)}


def fmt(ms, nd=3):
    if not ms:
        return "—"
    s = f"{ms['mean']:.{nd}f}"
    if ms["se"] is not None:
        s += f"±{ms['se']:.{nd}f}"
    return s + f" (n={ms['n']})"


def primary_value(rs, primary):
    if rs.get("heldout") and rs["heldout"].get(primary) is not None:
        return rs["heldout"][primary], "heldout"
    if rs.get("selected"):
        return rs["selected"]["dev"].get(primary), "dev"
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="逗号分隔的 glob；含 run_meta.json+scores.jsonl 的目录即一个运行")
    ap.add_argument("--primary", default="creativity")
    ap.add_argument("--heldout_name", default="heldout")
    ap.add_argument("--label_by", choices=["meta", "dir"], default="meta")
    ap.add_argument("--ref", default=None, help="参照标签（如 ntp_fixed）")
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--min_seeds", type=int, default=3)
    ap.add_argument("--emit_selection", default=None, help="写出 run\\tstep\\tckpt_dir（供留出评测）")
    ap.add_argument("--selection", default=None,
                    help="读取冻结 run\\tstep\\tckpt_dir 清单；报告按清单步数而非重新选择")
    ap.add_argument("--overwrite_selection", action="store_true",
                    help="显式允许替换已有 selection manifest（默认拒绝，冻结清单只读）")
    ap.add_argument("--allow_duplicates", action="store_true",
                    help="允许同一标签下出现重复种子（默认报错：防止不同规格的运行混入同一标签）")
    ap.add_argument("--out", default=None, help="Markdown 报告")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    frozen_selection = load_selection(args.selection) if args.selection else {}
    runs = [summarize_run(r, args, frozen_selection) for r in discover(args.runs)]
    seen_ls = {}
    for rs in runs:
        seen_ls.setdefault((rs["label"], rs["seed"]), []).append(rs["run"])
    dups = {k: v for k, v in seen_ls.items() if len(v) > 1}
    if dups and not args.allow_duplicates:
        raise SystemExit("同一标签下有重复种子（多半是不同规格的运行共用了标签；改标签或用 --label_by dir）：\n" +
                         "\n".join(f"  {k}: {v}" for k, v in dups.items()))
    if args.emit_selection:
        if os.path.exists(args.emit_selection) and not args.overwrite_selection:
            raise SystemExit(
                f"selection manifest already exists: {args.emit_selection}; "
                "choose a new path or pass --overwrite_selection explicitly"
            )
        with open(args.emit_selection, "w", encoding="utf-8") as f:
            for rs in runs:
                if rs.get("selected") and rs["selected"]["ckpt_dir"]:
                    f.write(f"{rs['run']}\t{rs['selected']['step']}\t{rs['selected']['ckpt_dir']}\n")

    labels = sorted({rs["label"] for rs in runs})
    agg = {}
    for lab in labels:
        rr = [rs for rs in runs if rs["label"] == lab]
        P = args.primary
        agg[lab] = {
            "n_runs": len(rr), "seeds": sorted(rs["seed"] for rs in rr),
            "sel_step": [rs["selected"]["step"] if rs.get("selected") else None for rs in rr],
            "dev_sel": mean_se([rs["selected"]["dev"].get(P) if rs.get("selected") else None for rs in rr]),
            "heldout_sel": mean_se([(rs.get("heldout") or {}).get(P) for rs in rr]),
            "validity_sel": mean_se([(rs.get("heldout") or (rs.get("selected") or {}).get("dev") or {}).get("test_validity") for rs in rr]),
            "memo_own_sel": mean_se([((rs.get("selected") or {}).get("dev") or {}).get("memo_own") for rs in rr]),
            "endpoint": mean_se([(rs["traj"][P] or {}).get("endpoint") for rs in rr]),
            "peak": mean_se([(rs["traj"][P] or {}).get("peak") for rs in rr]),
            "auc": mean_se([(rs["traj"][P] or {}).get("auc_mean") for rs in rr]),
            "derived": any(rs["creativity_derived"] for rs in rr)}
    verdicts = {}
    if args.ref and args.ref in agg:
        ref = {rs["seed"]: rs for rs in runs if rs["label"] == args.ref}
        for lab in labels:
            if lab == args.ref:
                continue
            diffs, sources = [], set()
            for rs in runs:
                if rs["label"] != lab or rs["seed"] not in ref:
                    continue
                a, sa = primary_value(rs, args.primary)
                b, sb = primary_value(ref[rs["seed"]], args.primary)
                if a is None or b is None:
                    continue
                diffs.append(a - b)
                sources |= {sa, sb}
            if not diffs:
                verdicts[lab] = {"verdict": "NO_PAIRS"}
                continue
            d = np.array(diffs)
            ok_n = len(d) >= args.min_seeds
            heldout_only = sources == {"heldout"}
            passed = ok_n and heldout_only and d.mean() >= args.margin and (d > 0).all()
            verdicts[lab] = {"n_pairs": len(d), "mean_diff": float(d.mean()), "min_diff": float(d.min()),
                             "per_seed": [float(x) for x in d], "sources": sorted(sources),
                             "verdict": "PASS" if passed else
                             ("INSUFFICIENT" if not (ok_n and heldout_only) else "FAIL")}

    tasks = sorted({rs.get("task") for rs in runs if rs.get("task")})
    data_dirs = sorted({rs.get("data_dir") for rs in runs if rs.get("data_dir")})
    res = {"primary": args.primary, "ref": args.ref, "margin": args.margin,
           "min_seeds": args.min_seeds, "tasks": tasks, "data_dirs": data_dirs,
           "selection_file": display_path(args.selection) if args.selection else None,
           "selection_sha256": sha256_file(args.selection),
           "aggregate": agg, "verdicts": verdicts, "runs": runs}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

    task_note = ", ".join(tasks) if tasks else "未记录"
    lines = [f"# 汇总（主指标 {args.primary}；参照 {args.ref or '—'}；margin {args.margin}；"
             f"min_seeds {args.min_seeds}）", "", f"评价任务：{task_note}", "",
             "| 标签 | 种子 | 选中步 | 开发集@选中 | 留出集@选中 | validity@选中 | memo_own@选中 | 终点 | 峰值 | 轨迹均值 | Δ vs 参照（均值 / 最小） | 判定 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab in labels:
        a = agg[lab]
        v = verdicts.get(lab, {})
        dv = f"{v['mean_diff']:+.3f} / {v['min_diff']:+.3f} ({'+'.join(v['sources'])})" if "mean_diff" in v else "—"
        lines.append(f"| {lab}{' †' if a['derived'] else ''} | {','.join(map(str, a['seeds']))} | "
                     f"{','.join(str(s) for s in a['sel_step'])} | {fmt(a['dev_sel'])} | {fmt(a['heldout_sel'])} | "
                     f"{fmt(a['validity_sel'])} | {fmt(a['memo_own_sel'])} | {fmt(a['endpoint'])} | "
                     f"{fmt(a['peak'])} | {fmt(a['auc'])} | {dv} | {v.get('verdict', '—')} |")
    lines += ["", "† creativity 由旧 scores 的 n_unique_valid_novel / 测试探针数推得（事件 #4 修复前的运行仍受其影响）。",
              "开发集峰值是对多个检查点取最大，系统性高估；判定只用留出集数值。", "",
              "## 逐运行", "", "| 运行 | 标签 | 种子 | 选中步 | 开发集@选中 | 留出集@选中 | 终点 | 峰值@步 |", "|---|---|---|---|---|---|---|---|"]
    for rs in runs:
        t = rs["traj"][args.primary] or {}
        sel = rs.get("selected") or {}
        h = (rs.get("heldout") or {}).get(args.primary)
        lines.append(f"| {rs['run']} | {rs['label']} | {rs['seed']} | {sel.get('step', '—')} | "
                     f"{(sel.get('dev') or {}).get(args.primary, float('nan')):.3f} | "
                     f"{'—' if h is None else f'{h:.3f}'} | {t.get('endpoint', float('nan')):.3f} | "
                     f"{t.get('peak', float('nan')):.3f}@{t.get('peak_step', '—')} |")
    md = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
    # Windows PowerShell can expose a GBK stdout even though report files are
    # UTF-8.  Keep the CLI/report tests usable without requiring callers to
    # set PYTHONIOENCODING explicitly.
    try:
        print(md)
    except UnicodeEncodeError:
        stream = getattr(sys.stdout, "buffer", None)
        if stream is None:
            raise
        stream.write(md.encode("utf-8"))


if __name__ == "__main__":
    main()
