# Graph-variant tool verification

Date: 2026-10-17  
Scope: engineering verification only; no graph-factorial training run was started.

## Source and checks

- Source commit: `1274bde` (local cherry-pick of remote change `48ee55d`, whose parent is the current Sibling launch commit `2710cee`).
- Code checked: `code/make_variant_data.py`, `code/test_pipeline.py`.
- Static check: `python -m py_compile` passed.
- Synthetic graph checks passed for `(1,1)`, `(1,2)`, `(1,4)`, `(2,2)`, and a fresh-hash cell. The checks covered degree counts, edge counts, excluded-hash disjointness, and metadata.
- Full pipeline test: on `gpu-gfkd`, in an isolated `/tmp/p6-test-1274bde` checkout, running the existing Python environment's `code/test_pipeline.py` completed with `ALL OK` (39 tests).
- Manifest check: the existing development Sibling manifest still validates with `validate_sibling_manifest.py`; it records nine blocked runs and no training start in that manifest.

## Interpretation boundary

These checks establish that the graph variant generator and its guards are executable and regression-tested. They are not evidence for a scientific effect, and they do not validate any graph-factorial result. Formal graph runs require a separately frozen addendum, fresh output directories, and a new held-out evaluation record.
