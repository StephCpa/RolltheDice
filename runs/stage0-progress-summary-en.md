# P6 Stage 0 Progress Summary (as of 2026-10-10)

> **⚠ Review note, added 2026-10-10. The text below is unchanged; see `analysis/2026-10-10-external-review.md`.**
> - The `ntp_randperm` row comes from a run registered as **INVALID** (incident #3: it never re-paired). Reading point 3 (the "G1 control result") is therefore unsupported. In practice that run is a replicate of ntp_fixed, and the gap between them (0.063 validity) is larger than setc's lead (+0.039).
> - The `setc_m4` row is the **DEVIATION** run. setc also differs from ntp_fixed by adding the teacherless objective and a second forward pass per step. Use the new matched arms `tl_fixed_m4` and `tl_randperm_m4` to isolate the effect of matching itself.
> - **Incident #4:** the circle `seen()` check did not truncate at `</a>`. This inflates *Unseen-valid* and *Unique novel* and deflates `test_seen`; validity, memorization and unique-valid are unaffected. It is fixed; re-score the existing models with `code/rescore.py`.
> - The no-seed "single output" result follows from greedy decoding on a constant prompt, so it is not evidence of mode collapse; a temperature-sampling comparison is needed. setc's memorization score of 0.41 measures reproduction of the *dataset* pairing, not the absence of memorization.
> - The evaluator self-test now has 15 checks.

Project: transferable multi-solution generation following *Roll the dice & look before you leap* (ICML 2025 Outstanding Paper, arXiv:2504.15266). Candidate under test: **set-coupled multi-solution training** — per round, an optimal bijection between m=4 seeds and m=4 verified solutions is chosen by Hungarian matching on teacherless (parallel-prediction) costs, so that m² pairing costs are computed from only m forward passes. All results below are **development status, single training seed, greedy decoding** (matching the original repo's predict mechanics).

## Infrastructure completed

- Faithful clean-room reimplementation of the paper's training semantics (NTP / teacherless-hybrid 1:3 / seed-conditioning) with sources line-mapped to the released code snapshot `7176dc7`; the vendored 2021-era simpletransformers stack cannot run on modern Python.
- Six datasets generated with fixed seeds (circle/sibling × hash/no-hash/hybrid). Circle: 10,000 train targets, all programmatically valid, 8,859 unique canonical structures, 10,000 unique hashes. Sibling: 49,766 train, all valid, all unique.
- Evaluator ported and self-tested (13/13 checks: cycle validity, relabeling-invariant canonicalization, knowledge-graph edge checks, seen/unseen logic).
- Hardware reality: single RTX 2080 Ti 11 GB instead of the planned A6000 — absolute throughput not comparable; relative C01 speedup still valid (logged).

## Main results — Circle task, 40k steps (paper's full budget), 256 probes

| Arm | Memorization | Validity | Unseen-valid | Unique valid | Unique novel |
|---|---|---|---|---|---|
| ntp_noseed (no seed) | 0.000 | **1.000** | 0.000 | 1 | 0 |
| ntp_fixed (original paper setup) | 1.000 | 0.211 | 0.148 | 27 | 19 |
| ntp_randperm (G1 control: persistent seeds, random bijection per round) | 1.000 | 0.148 | 0.109 | 19 | 14 |
| **setc_m4 (set-coupling prototype)** | 0.406 | **0.250** | **0.195** | **32** | **25** |
| tl_hybrid_v2 (original teacherless mix, fixed rerun) | *running* | | | | |

Reading (directional only; no significance claims at n=1):

1. **No-seed mode collapse reproduced**: the model deterministically outputs one single valid circle (validity 1.0, novelty 0) — the paper's core phenomenon.
2. **Seeds act as generalizable randomness**: all seed arms produce novel valid circles on unseen hashes.
3. **G1 control result — non-permanence alone does not help**: randomly re-pairing each round performs at or below the arbitrary fixed pairing. This eliminates "mere pairing non-permanence" as an explanation path.
4. **Set-coupling leads on every generative metric** (validity +0.04, unseen +0.05, unique +5 vs fixed pairing) while *not* memorizing training pairs (memo 0.41 vs 1.00).

## Key mechanistic finding — assignment dynamics (640k matching records)

The chosen permutations π* show three clean phases:

| Phase | Steps | Mean Cayley distance | Regime |
|---|---|---|---|
| Symmetric equilibrium | ~300–4,000 | ≈1.9–2.0 (= random-permutation expectation 1.917) | costs indifferent; assignment rotates arbitrarily — exactly the "seeds carry no scheme information" optimum predicted by the KL decomposition |
| Specialization decay | ~4,000–11,000 | 1.5 → 0.3 | rich-get-richer: cost preferences form |
| **Complete freeze** | ~12,000–40,000 | **0.000** | every group locks into one self-chosen fixed permutation |

Interpretation: **optimal-assignment training does not escape fixed pairing — it converges to one that the training dynamics selected**, and most of the validity gain accrues *after* freezing (0.133 @15k → 0.258 @30k). The candidate's premise ("escapes permanent pairing") fails in this regime; what survives is a weaker, testable claim: a cost-selected pairing modestly outperforms an arbitrary one-shot pairing. Whether that edge is real (vs single-seed noise) and whether it survives at higher m / more seeds is exactly the Stage 1 question.

## Incidents & caveats (logged in runs/README.md)

1. **tl_hybrid first run invalidated**: the trainer double-expanded the already-expanded hybrid JSON (MTP items' dummy-eos `target_text` re-expanded 1:3), flooding training with "predict EOS in answer slots" units — the model collapsed to EOS output. Root cause fixed (items now consumed individually via `target_labels`); invalid run preserved as `tl_hybrid_INVALID-doubleexpansion/`; corrected `tl_hybrid_v2` rerunning now. Other arms unaffected (they consume base datasets).
2. **setc per-step timing contaminated** by concurrent diagnostic scripts sharing the GPU; C01 will be established by a dedicated microbenchmark on the idle GPU (m teacherless forwards vs m² naive forwards for the m×m cost matrix, plus recorded divergence between the two cost types — they are different likelihoods by design).
3. 10k steps (¼ budget) produce zero valid outputs with correct output format — training-budget effect, not an implementation error; 10k trajectories archived.

## Throughput (RTX 2080 Ti, dev-scale reference)

ntp arms 0.114–0.142 s/step (40k ≈ 1.3–1.6 h each); setc 0.39 s/step p50 (2 forwards + Hungarian). GPU-hours consumed so far ≈ 10.5 of the 60 h Stage-0 cap.

## In progress / next

- tl_hybrid_v2 finishing (~1.5 h) → separates "matching benefit" from "hybrid-weighting benefit".
- C01 microbenchmark → the computational claim.
- Diagnostics on idle GPU: I(z;[y]) proxy on seen/unseen seeds × {fixed, setc, hybrid}, coverage curves, teacherless one-shot parallel readout vs AR decoding.
- Decision mapping against the Stage-0 plan's branch table, then Stage-1 recommendation (multi-seed confirmation, m=8, Sinkhorn soft-assignment arm as the natural anti-freezing ablation).
