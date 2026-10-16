"""Create explicit exposure-control data variants without changing a base snapshot.

The ``repeat_rows`` variant keeps the same hash->answer incidence graph and
duplicates every training row ``k`` times.  It is a development control for
the 2hash condition: it matches the number of training rows while adding no
new hash support and no new edges.  Evaluation splits are copied byte-for-byte.

Example::

    python code/make_exposure_variant.py --base_dir data/circle... \
        --out data/circle...repeat2 --k 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_key(row: dict) -> str:
    return row["input_text"].split("<q>", 1)[0]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args(argv)
    if args.k < 2:
        ap.error("--k must be at least 2")

    base = Path(args.base_dir)
    out = Path(args.out)
    if not (base / "train.json").is_file():
        ap.error(f"missing train.json under {base}")
    if out.exists() and any(out.iterdir()):
        ap.error(f"output directory is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    train_path = base / "train.json"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    if not train:
        ap.error("train.json is empty")
    keys = [hash_key(row) for row in train]
    if len(keys) != len(set(keys)):
        ap.error("base train.json must have unique hashes for repeat_rows")

    out_train = train * args.k
    (out / "train.json").write_text(
        json.dumps(out_train, ensure_ascii=False), encoding="utf-8"
    )
    copied = []
    for name in ("valid.json", "test.json", "vocab.json", "heldout.json"):
        src = base / name
        if src.is_file():
            shutil.copyfile(src, out / name)
            copied.append(name)

    manifest = {
        "schema_version": "p6-exposure-variant-v1",
        "mode": "repeat_rows",
        "k": args.k,
        "base_dir": str(base),
        "base_train_sha256": sha256_file(train_path),
        "train_sha256": sha256_file(out / "train.json"),
        "n_train_base": len(train),
        "n_train_out": len(out_train),
        "n_unique_hash_base": len(set(keys)),
        "n_unique_hash_out": len(set(hash_key(row) for row in out_train)),
        "unique_edge_count_base": len(train),
        "unique_edge_count_out": len(set((hash_key(row), row["target_text"]) for row in out_train)),
        "copied_splits": copied,
        "formal_use": "development_only_until_preregistered",
    }
    (out / "variant.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
