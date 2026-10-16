"""Re-score a saved checkpoint without training.

The command is deliberately strict about provenance. A re-score must name a
task explicitly and that task must agree with the run metadata. The selected
checkpoint is identified from its directory (``ckpts/stepN`` or
``final_model``), rather than from a free-form ``--step`` label. Existing
outputs are reusable only when their task, checkpoint, evaluator, and probe
identities all match; legacy or conflicting output is never overwritten.
"""
import argparse
import hashlib
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluator import EVALUATOR_VERSION, SIBLING_GRAPH_SCHEMA_VERSION, load_sibling_graph  # noqa: E402
from train import load_pairing, run_eval  # noqa: E402


TASKS = ("circle", "sibling", "line")
META_SCHEMA_VERSION = 2


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(path):
    return os.path.realpath(os.path.abspath(path))


def find_run_meta(model_dir):
    """Find the nearest run_meta.json above a checkpoint directory."""
    cur = canonical(model_dir)
    while True:
        p = os.path.join(cur, "run_meta.json")
        if os.path.isfile(p):
            return p
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def load_run_meta(model_dir):
    path = find_run_meta(model_dir)
    if path is None:
        raise SystemExit(
            f"[error] no run_meta.json found above --model_dir {model_dir}; "
            "task/checkpoint identity cannot be verified"
        )
    try:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"[error] cannot read run metadata {path}: {e}") from e
    task = meta.get("task")
    if task not in TASKS:
        raise SystemExit(f"[error] run metadata {path} has missing/unknown task {task!r}")
    return path, meta


def assert_task_matches(task, run_meta_path, run_meta):
    if run_meta.get("task") != task:
        raise SystemExit(
            f"[error] task conflict: --task {task!r} vs {run_meta_path} task "
            f"{run_meta.get('task')!r}"
        )


def checkpoint_identity(model_dir, step, run_meta):
    """Return an identity and reject misleading ``--step`` labels."""
    md = canonical(model_dir)
    if not os.path.isdir(md):
        raise SystemExit(f"[error] checkpoint directory does not exist: {model_dir}")
    base = os.path.basename(md)
    m = re.fullmatch(r"step(\d+)", base)
    if m:
        if os.path.basename(os.path.dirname(md)) != "ckpts":
            raise SystemExit(
                f"[error] checkpoint directory {md} is not under a ckpts/ directory"
            )
        actual = int(m.group(1))
        if step != actual:
            raise SystemExit(
                f"[error] --step {step} conflicts with checkpoint directory {base}; "
                "--step is a label only when it agrees with ckpts/stepN"
            )
        return {"kind": "checkpoint", "step": actual, "path": md}
    if base == "final_model":
        final_step = run_meta.get("steps")
        if not isinstance(final_step, int):
            raise SystemExit(
                "[error] final_model requires integer run_meta.steps to identify its checkpoint"
            )
        if step != final_step:
            raise SystemExit(
                f"[error] --step {step} conflicts with final_model run endpoint {final_step}; "
                "use the selected checkpoint directory"
            )
        return {"kind": "final_model", "step": final_step, "path": md}
    raise SystemExit(
        f"[error] unrecognized model directory {md}; use .../ckpts/stepN or .../final_model"
    )


def probe_path(data_dir, probe_file):
    return canonical(probe_file) if probe_file else canonical(os.path.join(data_dir, "test.json"))


def build_identity(args, model_dir, run_meta_path, run_meta, probe):
    ev_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluator.py")
    model_id = checkpoint_identity(model_dir, args.step, run_meta)
    ppath = probe_path(args.data_dir, args.probe_file)
    if not os.path.isfile(ppath):
        raise SystemExit(f"[error] probe file does not exist: {ppath}")
    data_train = os.path.join(args.data_dir, "train.json")
    graph_meta = None
    if args.task == "sibling":
        if not args.sibling_graph_dir:
            raise SystemExit("[error] task=sibling requires --sibling_graph_dir")
        try:
            load_sibling_graph(args.sibling_graph_dir)
        except ValueError as exc:
            raise SystemExit(f"[error] {exc}") from exc
        graph_meta = {
            "schema_version": SIBLING_GRAPH_SCHEMA_VERSION,
            "dir": canonical(args.sibling_graph_dir),
            "files": {name: sha256_file(os.path.join(args.sibling_graph_dir, name))
                      for name in ("entities_b1_dict.json", "entities_b2_dict.json")},
        }
    return {
        "schema_version": META_SCHEMA_VERSION,
        "task": args.task,
        "step": args.step,
        "model_dir": canonical(model_dir),
        "checkpoint_identity": model_id,
        "run_meta_path": canonical(run_meta_path),
        "run_meta_sha256": sha256_file(run_meta_path),
        "data_dir": canonical(args.data_dir),
        "data_train_sha256": sha256_file(data_train) if os.path.isfile(data_train) else None,
        "probe_file": ppath,
        "probe_sha256": sha256_file(ppath),
        "probe_kind": "explicit" if args.probe_file else "data_dir/test.json",
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_sha256": sha256_file(ev_path),
        "n_train_probes": len(probe[0]),
        "n_test_probes": len(probe[1]),
        "sibling_graph": graph_meta,
    }


def assert_metadata_compatible(out_dir, expected):
    """Refuse to overwrite any existing result with unknown provenance."""
    meta_path = os.path.join(out_dir, "rescore_meta.json")
    scores_path = os.path.join(out_dir, "scores.jsonl")
    contents = os.listdir(out_dir) if os.path.isdir(out_dir) else []
    if not os.path.exists(meta_path) and not os.path.exists(scores_path) and not contents:
        return False
    if not os.path.isfile(meta_path):
        raise SystemExit(
            f"[error] existing {out_dir} has result artifacts but no rescore_meta.json; "
            "legacy output cannot be reused or overwritten (archive it and choose a new output dir)"
        )
    try:
        with open(meta_path, encoding="utf-8") as f:
            got = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"[error] cannot read existing provenance {meta_path}: {e}") from e
    mismatches = []
    for key, value in expected.items():
        if got.get(key) != value:
            mismatches.append(f"{key}: existing={got.get(key)!r}, requested={value!r}")
    if mismatches:
        raise SystemExit(
            f"[error] existing result provenance conflicts in {out_dir}; refusing reuse/overwrite\n"
            + "\n".join("  " + x for x in mismatches)
        )
    if os.path.isfile(scores_path):
        print(f"[skip] compatible rescore already exists: {scores_path}")
        return True
    return False


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--task", choices=TASKS, required=True,
                    help="任务身份；必须与最近的 run_meta.json 一致")
    ap.add_argument("--sibling_graph_dir", default=None,
                    help="完整 Sibling graph JSON 目录；task=sibling 时必填")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe_file", default=None,
                    help="测试探针来源（缺省 data_dir/test.json 的 type=test 条目）")
    ap.add_argument("--n_test", type=int, default=None, help="测试探针数（缺省=来源中全部）")
    ap.add_argument("--n_train", type=int, default=128, help="训练探针数（取自 test.json 的 type=train）")
    ap.add_argument("--train_offset", type=int, default=0, help="训练探针起始偏移（避开开发集用 128）")
    ap.add_argument("--n_probes", type=int, default=None,
                    help="旧接口：train/test 各取一半（给定时覆盖 --n_test/--n_train）")
    ap.add_argument("--pairing_file", default=None,
                    help="本臂实际训练配对（memo_own）；缺省 model_dir/pairing.json，再缺省名义")
    ap.add_argument("--n_samples", type=int, default=1, help="每探针生成数")
    ap.add_argument("--sample", action="store_true", help="温度采样（默认 greedy）")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--step", type=int, required=True,
                    help="选择标签；必须与 ckpts/stepN 或 run_meta.steps 一致")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.task == "sibling" and not args.sibling_graph_dir:
        ap.error("task=sibling requires --sibling_graph_dir with both complete graph JSON files")
    if args.task != "sibling" and args.sibling_graph_dir:
        ap.error("--sibling_graph_dir is only valid with task=sibling")
    return args


def main(argv=None):
    args = parse_args(argv)
    run_meta_path, run_meta = load_run_meta(args.model_dir)
    assert_task_matches(args.task, run_meta_path, run_meta)

    with open(os.path.join(args.data_dir, "train.json"), encoding="utf-8") as f:
        raw = json.load(f)
    with open(os.path.join(args.data_dir, "test.json"), encoding="utf-8") as f:
        probes_all = json.load(f)
    src = probes_all
    if args.probe_file:
        with open(args.probe_file, encoding="utf-8") as f:
            src = json.load(f)
    n_test, n_train = args.n_test, args.n_train
    if args.n_probes is not None:
        n_test = n_train = args.n_probes // 2
    te = [p for p in src if p.get("type", "test") == "test"]
    te = te[:n_test] if n_test is not None else te
    tr = [p for p in probes_all if p.get("type") == "train"][args.train_offset:args.train_offset + n_train]
    identity = build_identity(args, args.model_dir, run_meta_path, run_meta, (tr, te))
    os.makedirs(args.out, exist_ok=True)
    if assert_metadata_compatible(args.out, identity):
        return 0

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained(args.model_dir)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model_dir).to(device)

    sibling_graph = None
    if args.task == "sibling":
        sibling_graph = load_sibling_graph(args.sibling_graph_dir)

    pf = args.pairing_file
    if pf is None and os.path.exists(os.path.join(args.model_dir, "pairing.json")):
        pf = os.path.join(args.model_dir, "pairing.json")
    text2idx = {s["input_text"]: k for k, s in enumerate(raw)}
    own_targets = None
    if len(text2idx) == len(raw):   # 无 hash 数据没有逐 seed 配对
        try:
            R = load_pairing(pf, len(raw)) if pf else list(range(len(raw)))
        except AssertionError as e:  # 配对来自另一份训练数据（如 dup_hash 变体）
            print(f"[warn] 配对与 --data_dir 不匹配，跳过 memo_own：{e}")
            R = None
        if R is not None:
            own_targets = {p["input_text"]: raw[R[text2idx[p["input_text"]]]]["target_text"]
                           for p in tr if p["input_text"] in text2idx}

    # Keep canonical identity fields authoritative even when callers used
    # relative paths in the CLI arguments.
    metadata = {**vars(args), **identity, "pairing_used": pf or "nominal"}
    with open(os.path.join(args.out, "rescore_meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    run_eval(model, tok, tr + te, raw, args.task, args.out, args.step, device,
             args.n_samples, args.temp, tok.eos_token_id, save_preds=True,
             do_sample=args.sample, own_targets=own_targets,
             extra={"model_dir": canonical(args.model_dir),
                    "probe_file": identity["probe_file"],
                    "task": args.task,
                    "checkpoint_identity": identity["checkpoint_identity"],
                    "evaluator_version": EVALUATOR_VERSION},
             sibling_graph=sibling_graph)
    return 0


if __name__ == "__main__":
    main()
