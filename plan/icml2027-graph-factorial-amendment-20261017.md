# ICML 2027 graph-factorial amendment (development)

Date: 2026-10-17  
Status: development planning record; no graph-factorial training run has started.

This amendment supersedes the proposed `repeat_rows_k2` Gate B in
`plan/icml2027-addendum-20261017.md`. The trainer samples rows uniformly with
`shuffle=True`, so duplicating every row leaves the sampling distribution the
same as a larger batch. `repeat_rows_k2` would therefore be an exposure/batch
control already represented by `ntp_fixed_b128`, not an independent
identification of the 2hash effect.

## Frozen development design

After the current Sibling Gate A has reached a stopping point, generate fresh
timestamped output directories for Circle and Line using
`code/make_variant_data.py --mode graph`. Every new cell uses the plain
`ntp_fixed` objective, batch size 128, 40,000 optimizer steps, and training
seeds 42, 43, and 44. The existing `b128` and `2hash` runs remain reference
cells; they are not silently relabelled or rerun as part of this amendment.

| Cell | Hashes | Answers | Hash degree d_H | Answer degree d_Y | Role |
|---|---:|---:|---:|---:|---|
| `G12` | 10,000 | 5,000 | 1 | 2 | equal hash count and rows; answer-side multiplicity |
| `G14` | 10,000 | 2,500 | 1 | 4 | answer-side multiplicity dose |
| `G22` | 10,000 | 10,000 | 2 | 2 | target-side multiplicity at fixed row count |
| `fresh20k` | 20,000 | 20,000 | 1 | 1 | more hashes/rows without multiplicity |

The four new cells × three seeds × two tasks are 24 runs. This is a
development budget estimate, not evidence of a result. The `2hash` reference
cell is the 20,000-hash, 10,000-answer, `(d_H,d_Y)=(1,2)` configuration.

## Evaluation and identification

Use the same checkpoint-selection and fresh-heldout protocol as the relevant
task's frozen evaluation contract. Record task identity, graph/data SHA256,
model revision, checkpoint identity, probe identity, optimizer steps, rows,
unique hashes, unique answers, unique edges, and per-edge exposure. A missing
or conflicting task identity must fail closed.

For `G12` and `G14`, validity and `test_seen` are co-primary diagnostics
because reducing the training answer set can mechanically increase the novel
share. Report canonical creativity, raw creativity, validity, unique novel,
canonical `test_seen`, and memorization for every cell, with per-seed values
and paired differences. Do not treat a PASS/FAIL gate as a substitute for
the continuous effects.

The predeclared readings are:

1. `G12` versus `b128`: answer-side multiplicity at equal hash count and row
   count.
2. `fresh20k` versus `b128`: more distinct hashes and rows without
   multiplicity.
3. `2hash` versus `fresh20k`: multiplicity at equal hash count and row count.
4. `G14` versus `G12`: a dose response in answer-side multiplicity.
5. `G22` versus `G12`: adding target-side multiplicity while holding answer
   count fixed.

These comparisons are hypotheses to be tested after launch. They do not
authorize changing the margin, selecting a favorable cell, or adding more
degrees during the run. If the answer-set-size confound remains material,
state that limitation rather than converting the cells into a causal claim.

## Launch guard

Before any formal run, freeze a task-specific manifest containing the input
snapshot, answer pool, exclusions, graph metadata, model revision, seed list,
step/batch settings, output root, and evaluation version. Keep graph data and
results append-only under new output paths. The engineering verification in
`outputs/sibling-confirmation-20261016/graph-variant-verification-20261017.md`
does not count as a scientific result.
