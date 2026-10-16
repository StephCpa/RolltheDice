# Sibling derived-data candidate (development)

Date: 2026-10-17 (Asia/Shanghai)

This record is a preparation artifact. It does not unlock formal Sibling
training or claim that the upstream 5x500 artifact has been reproduced.

## Source and preserved inputs

- Dataset: `ChenWu98/sibling.5.500.10.50000`
- Dataset revision: `edb56d3aba44007a926f1f3535812009b28934e4`
- Public train parquet SHA-256:
  `2e12567550368a892f56b240bdfb6d819dee93ef2b0f27aa7b96a9b00c6d6ffb`
- Public valid parquet SHA-256:
  `3c252eb8d1e6243d6826fb307cd887d7798893e3a509c9ce73377369829ff1c3`
- Preserved rows: train 50,000; valid 3,000.

The public dataset does not contain `test.json`, `heldout.json`,
`entities_b1_dict.json`, or `entities_b2_dict.json`.

## Derived material

- Graph: the contiguous `a_i` to `b1_j`/`b2_j` block rule in the upstream
  `sibling.ipynb`, applied to `num_a=5`, `num_b_per_a=500`. The graph is
  explicitly **derived**, not an official file from the upstream release.
- Development probes: 3,000 train rows sampled with seed `20261017`, tagged
  `type=train`, plus all 3,000 public valid rows tagged `type=test`.
- `dup_hash(k=2)` variant: generated from the preserved train rows with the
  existing `make_variant_data.py`, variant seed `20261012`; new hashes are
  disjoint from train/valid/test in the candidate package.

The derived test file is a development probe contract. It is not an
independent heldout set. A formal run requires a new heldout hash set generated
after the data contract is accepted and frozen.

## Gate status

`candidate_manifest.json` records file-level hashes and the complete derivation
metadata. Its current `formal_use` is
`prohibited_until_owner_accepts_derived_contract`. The main freeze manifest
remains `blocked_pending_artifacts`; no Sibling training, checkpoint, or
heldout evaluation was started.

The manifest-driven entry point is `code/run_sibling_confirmation.py`. It can
produce a dry-run command for one exact arm/seed. Formal start requires the
freeze status and all artifact gates to be explicitly changed; it never
auto-selects GPUs or creates extra arms.

The training metadata now records a model revision label plus content hashes
for the serialized model config and tokenizer. Formal start will require that
revision label instead of relying on the mutable `gpt2` name alone.
