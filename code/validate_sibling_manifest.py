"""Validate the development-only Sibling confirmation launch manifest.

This is a structural preflight.  It deliberately does not inspect model
quality, run training, or infer a scientific result from the manifest.
"""
import argparse
import csv
import json
from pathlib import Path


EXPECTED_ARMS = {
    "ntp_fixed": 64,
    "ntp_fixed_b128": 128,
    "ntp_fixed_2hash": 128,
}
EXPECTED_SEEDS = {42, 43, 44}


def validate(manifest_path: Path, tsv_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    if manifest.get("status") != "blocked_pending_artifacts":
        errors.append("manifest must remain blocked_pending_artifacts before launch")
    if manifest.get("formal_run_started") or manifest.get("formal_results_present"):
        errors.append("formal run/result flags must be false")
    if manifest.get("scale_started"):
        errors.append("scale_started must be false")
    if manifest.get("protocol", {}).get("task") != "sibling":
        errors.append("protocol.task must be sibling")
    if manifest.get("protocol", {}).get("planned_steps", 0) <= 0:
        errors.append("planned_steps must be positive")

    runs = manifest.get("run_manifest", [])
    pairs = {(row.get("arm"), row.get("seed")) for row in runs}
    expected = {(arm, seed) for arm in EXPECTED_ARMS for seed in EXPECTED_SEEDS}
    if pairs != expected or len(runs) != 9:
        errors.append("run_manifest must contain exactly three arms x three seeds")
    for row in runs:
        arm = row.get("arm")
        if arm not in EXPECTED_ARMS:
            errors.append(f"unexpected arm: {arm}")
        elif row.get("seed") in EXPECTED_SEEDS and row.get("batch_sequences") not in (None, EXPECTED_ARMS[arm]):
            errors.append(f"batch mismatch for {arm}/{row.get('seed')}")

    with tsv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    tsv_pairs = {(row["arm"], int(row["seed"])) for row in rows}
    if tsv_pairs != expected or len(rows) != 9:
        errors.append("run_manifest.tsv must contain exactly three arms x three seeds")
    for row in rows:
        arm = row["arm"]
        if arm not in EXPECTED_ARMS:
            errors.append(f"unexpected TSV arm: {arm}")
        elif int(row["batch_sequences"]) != EXPECTED_ARMS[arm]:
            errors.append(f"TSV batch mismatch for {arm}/{row['seed']}")
        if row.get("formal_status") != "blocked_pending_artifacts":
            errors.append(f"TSV status is not blocked for {arm}/{row['seed']}")

    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    print(f"OK: development Sibling manifest validated ({len(runs)} blocked runs; no training started)")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("tsv", type=Path)
    args = parser.parse_args(argv)
    validate(args.manifest, args.tsv)


if __name__ == "__main__":
    main()
