"""Materialize a development-only Sibling candidate from the public 5x500 artifact.

The public Hugging Face example contains train/valid parquet files only.  This
tool preserves those rows, derives a dev probe file from a frozen sample of
train plus the public valid split, and derives the complete block graph from
the upstream notebook's explicit 5x500 indexing rule.  It never labels the
derived files as an official upstream artifact and refuses to overwrite an
existing output directory.

This is a preparation tool.  Its output is not formal evidence until the
project owner accepts the derived test/graph contract and updates the frozen
manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path


INPUT_RE = re.compile(r"^(?P<hash>[a-z0-9]{10})<q>$")
TARGET_RE = re.compile(
    r"^(?P<hash>[a-z0-9]{10})<q>(?P<b1><b1_[0-9]+>)(?P<b2><b2_[0-9]+>)"
    r"(?P<a><a_[0-9]+>)</a>$"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path: Path, value: object) -> str:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, separators=(",", ":"))
    return sha256(path)


def read_parquet(path: Path) -> list[dict[str, str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("pyarrow is required to materialize the public parquet candidate") from exc
    table = pq.read_table(path, columns=["input_text", "target_text"])
    rows = table.to_pylist()
    return [{"input_text": str(r["input_text"]), "target_text": str(r["target_text"])} for r in rows]


def validate_rows(rows: list[dict[str, str]], label: str, *, placeholders: bool = False) -> None:
    if not rows:
        raise ValueError(f"{label} is empty")
    seen = set()
    for row in rows:
        im = INPUT_RE.fullmatch(row["input_text"])
        tm = TARGET_RE.fullmatch(row["target_text"])
        placeholder = re.fullmatch(r"[a-z0-9]{10}<q></a>", row["target_text"])
        valid_target = (tm is not None and im.group("hash") == tm.group("hash")) or (placeholders and placeholder)
        if not im or not valid_target:
            raise ValueError(f"invalid {label} row: {row!r}")
        seen.add(row["input_text"])
    if len(seen) != len(rows):
        raise ValueError(f"{label} contains duplicate hash inputs")


def graph(num_a: int, num_b_per_a: int) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    b1 = {f"<a_{i}>": [f"<b1_{i * num_b_per_a + j}>" for j in range(num_b_per_a)]
          for i in range(num_a)}
    b2 = {f"<a_{i}>": [f"<b2_{i * num_b_per_a + j}>" for j in range(num_b_per_a)]
          for i in range(num_a)}
    return b1, b2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", required=True, type=Path)
    ap.add_argument("--valid-parquet", required=True, type=Path)
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num-a", type=int, default=5)
    ap.add_argument("--num-b-per-a", type=int, default=500)
    ap.add_argument("--dev-train-probes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261017)
    args = ap.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    train = read_parquet(args.train_parquet)
    valid = read_parquet(args.valid_parquet)
    validate_rows(train, "train")
    validate_rows(valid, "valid", placeholders=True)
    if len(train) != 50000 or len(valid) != 3000:
        raise ValueError(f"unexpected public split sizes: train={len(train)}, valid={len(valid)}")

    input_hashes = {r["input_text"].split("<q>", 1)[0] for r in train + valid}
    rnd = random.Random(args.seed)
    if args.dev_train_probes > len(train):
        raise ValueError("dev-train-probes exceeds train rows")
    train_probe_idx = rnd.sample(range(len(train)), args.dev_train_probes)
    test = [{**train[i], "type": "train"} for i in train_probe_idx]
    test.extend({**row, "type": "test"} for row in valid)

    b1, b2 = graph(args.num_a, args.num_b_per_a)
    expected_a = {f"<a_{i}>" for i in range(args.num_a)}
    observed_a = {TARGET_RE.fullmatch(r["target_text"]).group("a") for r in train}
    if not observed_a <= expected_a:
        raise ValueError(f"observed answer entities outside derived graph: {sorted(observed_a - expected_a)}")

    data_dir = args.out / "base"
    graph_dir = args.out / "graph"
    data_dir.mkdir()
    graph_dir.mkdir()
    train_sha = dump_json(data_dir / "train.json", train)
    valid_sha = dump_json(data_dir / "valid.json", valid)
    test_sha = dump_json(data_dir / "test.json", test)
    vocab = ([f"<a_{i}>" for i in range(args.num_a)]
              + [f"<b1_{i}>" for i in range(args.num_a * args.num_b_per_a)]
              + [f"<b2_{i}>" for i in range(args.num_a * args.num_b_per_a)]
              + ["<mask>", "<sep>", "<a>", "</a>", "<q>", "</q>"])
    vocab_sha = dump_json(data_dir / "vocab.json", vocab)
    b1_sha = dump_json(graph_dir / "entities_b1_dict.json", b1)
    b2_sha = dump_json(graph_dir / "entities_b2_dict.json", b2)
    metadata = {
        "record_type": "development_sibling_public_candidate",
        "formal_use": "prohibited_until_owner_accepts_derived_contract",
        "source": {
            "dataset": "ChenWu98/sibling.5.500.10.50000",
            "revision": args.source_revision,
            "train_parquet": {"path": str(args.train_parquet), "sha256": sha256(args.train_parquet)},
            "valid_parquet": {"path": str(args.valid_parquet), "sha256": sha256(args.valid_parquet)},
        },
        "dimensions": {"num_a": args.num_a, "num_b_per_a": args.num_b_per_a, "hash_length": 10},
        "derived": {
            "test_policy": "3000 seeded train rows tagged train plus public valid rows tagged test",
            "test_seed": args.seed,
            "graph_policy": "contiguous block mapping from upstream sibling.ipynb applied to 5x500",
            "graph_is_official": False,
            "input_hashes_in_train_valid": len(input_hashes),
        },
        "files": {
            "base/train.json": {"rows": len(train), "sha256": train_sha},
            "base/valid.json": {"rows": len(valid), "sha256": valid_sha},
            "base/test.json": {"rows": len(test), "sha256": test_sha},
            "base/vocab.json": {"rows": len(vocab), "sha256": vocab_sha},
            "graph/entities_b1_dict.json": {"keys": len(b1), "sha256": b1_sha},
            "graph/entities_b2_dict.json": {"keys": len(b2), "sha256": b2_sha},
        },
    }
    with (args.out / "candidate_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
