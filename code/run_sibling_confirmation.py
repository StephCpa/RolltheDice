"""Manifest-driven launcher for one frozen Sibling confirmation run.

The launcher intentionally starts at most one row from run_manifest.tsv.  It
does not infer missing arms, generate a new selection table, or auto-select a
GPU.  ``--dry-run`` is safe while the proposal is blocked; ``--start`` only
works after the freeze manifest is explicitly moved to a ready state and all
identity inputs are supplied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


RUNS = {
    "ntp_fixed": ("base", 64),
    "ntp_fixed_b128": ("base", 128),
    "ntp_fixed_2hash": ("variant", 128),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def required_files(root: Path, *, graph: bool = False) -> list[Path]:
    names = (["entities_b1_dict.json", "entities_b2_dict.json"] if graph
             else ["train.json", "valid.json", "test.json", "vocab.json"])
    return [root / n for n in names]


def check_files(paths: list[Path], label: str) -> None:
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"{label} missing required files: {missing}")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def check_candidate(candidate: Path, base: Path, variant: Path, graph: Path, *, require_accepted: bool) -> dict:
    manifest_path = candidate / "candidate_manifest.json"
    check_files([manifest_path], "candidate")
    with manifest_path.open(encoding="utf-8") as f:
        cm = json.load(f)
    if require_accepted and cm.get("formal_use") != "accepted_by_owner":
        raise SystemExit(
            "candidate data is development-only; formal start requires "
            "candidate_manifest.formal_use=accepted_by_owner"
        )
    check_files(required_files(base), "base data")
    check_files(required_files(variant), "2hash variant")
    check_files(required_files(graph, graph=True), "Sibling graph")
    variant_meta = variant / "variant.json"
    check_files([variant_meta], "2hash variant metadata")
    with variant_meta.open(encoding="utf-8") as f:
        vm = json.load(f)
    if vm.get("mode") != "dup_hash" or vm.get("k") != 2:
        raise SystemExit(f"unexpected 2hash variant metadata: {vm}")
    expected = cm.get("files", {})
    actual_roots = {"base": base, "graph": graph}
    for rel, info in expected.items():
        top, name = rel.split("/", 1)
        path = actual_roots[top] / name
        actual = sha256(path)
        if actual != info.get("sha256"):
            raise SystemExit(f"candidate SHA mismatch for {rel}: {actual} != {info.get('sha256')}")
    base_rows = load_rows(base / "train.json")
    variant_rows = load_rows(variant / "train.json")
    if vm.get("n_train_base") != len(base_rows) or vm.get("n_train_out") != len(variant_rows):
        raise SystemExit(f"variant row-count mismatch: metadata={vm}, actual={len(base_rows)}/{len(variant_rows)}")
    base_hashes = {r["input_text"].split("<q>", 1)[0] for r in base_rows}
    variant_hashes = {r["input_text"].split("<q>", 1)[0] for r in variant_rows}
    if len(base_hashes) != len(base_rows) or len(variant_hashes) != len(variant_rows):
        raise SystemExit("base or variant train has duplicate hash inputs")
    if not base_hashes < variant_hashes or base_hashes & (variant_hashes - base_hashes):
        raise SystemExit("2hash variant is not a strict disjoint extension of base train hashes")
    if sha256(variant / "train.json") != vm.get("train_sha256"):
        raise SystemExit("variant.json train_sha256 does not match variant train.json")
    return cm


def read_run(manifest_path: Path, run_manifest_path: Path, arm: str, seed: int) -> tuple[dict, dict]:
    with manifest_path.open(encoding="utf-8") as f:
        fm = json.load(f)
    rows = []
    with run_manifest_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    hits = [r for r in rows if r.get("arm") == arm and int(r.get("seed", -1)) == seed]
    if len(hits) != 1:
        raise SystemExit(f"run manifest must contain exactly one row for {arm}/{seed}; got {len(hits)}")
    return fm, hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--run-manifest", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--base-data", required=True, type=Path)
    ap.add_argument("--variant-data", required=True, type=Path)
    ap.add_argument("--graph-dir", required=True, type=Path)
    ap.add_argument("--arm", choices=sorted(RUNS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--train-script", default=str(Path(__file__).with_name("train.py")))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--n-probes", type=int, default=256)
    ap.add_argument("--cuda-visible-devices", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--start", action="store_true")
    args = ap.parse_args()
    if args.dry_run == args.start:
        raise SystemExit("choose exactly one of --dry-run or --start")

    fm, row = read_run(args.manifest, args.run_manifest, args.arm, args.seed)
    candidate = check_candidate(
        args.candidate, args.base_data, args.variant_data, args.graph_dir,
        require_accepted=args.start,
    )
    if args.start:
        if fm.get("status") != "ready_for_formal_run" or fm.get("formal_run_started"):
            raise SystemExit(f"formal start refused by freeze manifest: status={fm.get('status')!r}")
        gates = fm.get("gates", {})
        bad = [k for k, v in gates.items() if v != "passed"]
        if bad:
            raise SystemExit(f"formal start refused; gates not passed: {bad}")

    mode, batch = RUNS[args.arm]
    data_dir = args.base_data if mode == "base" else args.variant_data
    check_files(required_files(data_dir), f"{args.arm} data")
    steps = args.steps if args.steps is not None else fm["protocol"]["planned_steps"]
    eval_every = args.eval_every if args.eval_every is not None else fm["protocol"]["eval_every"]
    ckpt_every = args.ckpt_every if args.ckpt_every is not None else fm["protocol"]["checkpoint_every"]
    out = args.out_root / f"{args.arm}__seed{args.seed}"
    cmd = [sys.executable, args.train_script, "--arm", "ntp_fixed", "--label", args.arm,
           "--data_dir", str(data_dir), "--task", "sibling", "--sibling_graph_dir", str(args.graph_dir),
           "--out", str(out), "--model", args.model, "--steps", str(steps), "--bs", str(batch),
           "--warmup", str(fm["protocol"]["warmup_steps"]), "--seed", str(args.seed),
           "--eval_every", str(eval_every), "--ckpt_every", str(ckpt_every),
           "--n_probes", str(args.n_probes), "--save_final"]
    summary = {
        "arm": args.arm,
        "seed": args.seed,
        "data_dir": str(data_dir),
        "graph_dir": str(args.graph_dir),
        "out": str(out),
        "data_files": {p.name: sha256(p) for p in required_files(data_dir)},
        "graph_files": {p.name: sha256(p) for p in required_files(args.graph_dir, graph=True)},
        "candidate_source_revision": candidate["source"]["revision"],
        "command": cmd,
        "cuda_visible_devices": args.cuda_visible_devices,
        "mode": "dry-run" if args.dry_run else "start",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
