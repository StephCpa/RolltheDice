"""Index evaluation provenance without loading models or regenerating outputs.

This is a read-only audit helper for archived runs.  It records what can be
verified from the checkout (run metadata, frozen selection, score metadata and
source hashes) and marks missing checkpoint/probe hashes as unavailable.  It
never upgrades a legacy score into a reusable formal result.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from datetime import date


def sha256_file(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(path: str) -> str:
    return os.path.realpath(os.path.abspath(path)).replace("\\", "/")


def load_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def last_score(path: str):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows[-1] if rows else None


def selection_rows(path: str):
    if not os.path.isfile(path):
        return {}
    result = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            run, step, checkpoint = line.rstrip("\n").split("\t")
            result[canonical(run)] = {
                "step": int(step),
                "checkpoint": canonical(checkpoint),
            }
    return result


def checkpoint_hash(path: str):
    """Return file hashes when weights are present; never hash directories."""
    if not os.path.isdir(path):
        return {"status": "missing", "files": {}}
    files = {}
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isfile(full) and (name.endswith((".bin", ".safetensors", ".json"))):
            files[name] = sha256_file(full)
    return {"status": "available" if files else "no_weight_files_in_checkout", "files": files}


def index_run(run: str, selection, evaluator_path: str):
    run = canonical(run)
    meta_path = os.path.join(run, "run_meta.json")
    score_path = os.path.join(run, "heldout", "scores.jsonl")
    meta = load_json(meta_path)
    score = last_score(score_path) if os.path.isfile(score_path) else None
    selected = selection.get(run)
    model_dir = (score or {}).get("model_dir")
    if model_dir:
        model_dir = canonical(model_dir)
    elif selected:
        model_dir = selected["checkpoint"]
    checkpoint = selected["checkpoint"] if selected else model_dir
    if selected and model_dir and model_dir != selected["checkpoint"]:
        checkpoint_status = "score_checkpoint_differs_from_selection"
    elif checkpoint:
        checkpoint_status = "matched_selection" if selected else "from_score_only"
    else:
        checkpoint_status = "missing"
    score_task = (score or {}).get("task")
    task_status = "verified" if score_task == meta.get("task") else "metadata_only"
    rescore_meta = os.path.join(run, "heldout", "rescore_meta.json")
    reusable = os.path.isfile(rescore_meta) and task_status == "verified"
    return {
        "run": run,
        "label": meta.get("label") or meta.get("arm"),
        "seed": meta.get("seed"),
        "task": meta.get("task"),
        "task_status": task_status,
        "run_meta_sha256": sha256_file(meta_path),
        "selected": selected,
        "score": {
            "step": (score or {}).get("step"),
            "creativity": (score or {}).get("creativity"),
            "test_validity": (score or {}).get("test_validity"),
            "model_dir": model_dir,
            "probe_file": (score or {}).get("probe_file"),
            "score_sha256": sha256_file(score_path),
        } if score else None,
        "checkpoint": {
            "path": checkpoint,
            "status": checkpoint_status,
            "hashes": checkpoint_hash(checkpoint) if checkpoint else {"status": "missing", "files": {}},
        },
        "rescore_meta": canonical(rescore_meta) if os.path.isfile(rescore_meta) else None,
        "reusable_formal_result": reusable,
        "limitation": None if reusable else "legacy score lacks task/evaluator/probe identity; retained for evidence, not formal reuse",
        "evaluator_source_sha256": sha256_file(evaluator_path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="glob of run directories, e.g. runs/line-phase1/*/s*")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    selection = selection_rows(args.selection)
    evaluator = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluator.py")
    runs = []
    for run in sorted(glob.glob(args.runs)):
        if os.path.isfile(os.path.join(run, "run_meta.json")):
            runs.append(index_run(run, selection, evaluator))
    output = {
        "schema_version": 1,
        "audit_date": date.today().isoformat(),
        "selection_file": canonical(args.selection),
        "selection_sha256": sha256_file(args.selection),
        "scope": "read-only index of existing Line evaluation records; no model or probe regeneration",
        "runs": runs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"runs": len(runs), "formal_reusable": sum(r["reusable_formal_result"] for r in runs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
