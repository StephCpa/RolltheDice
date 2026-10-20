# Review of the 2026-10-21 round: Sibling verdict, Circle factorial, and what the paper can still claim

Reviewer notes on the uploaded report (`report-en.md`, Sibling Gate A plus the Circle graph factorial). They are
written against `main` @ `6feaddb`. New tool: `code/coverage.py` (test `test_coverage_space_and_copy_ratio`).

---

## 1. Bottom line

- **The Sibling verdict stands as reported.** 2hash is below both b128 (−0.063) and fixed (−0.053). With that, no
  pairing scheme in this project improves creativity on all three tasks:
  - 2tgt fails on Line;
  - 2hash fails on Sibling;
  - G14 collapses on Circle.

  The "working recipe" framing in §Paper Claim Updates should therefore be dropped as well. There is no recipe left,
  only task-specific effects.
- **Part of the Circle factorial reading is a measurement artifact.** Circle has only 8! = 40,320 possible answer
  structures, so the training set's coverage of that space sets the chance rate at which valid outputs count as
  "seen" (§2). Corrected for coverage:
  - the **2hash − fresh20k = +0.047** creativity gap disappears (2hash 0.241 vs fresh20k 0.251 at equal coverage);
  - **G12 − b128 = +0.053** cannot be identified on creativity (G12 is 0.166–0.217 at equal coverage);
  - **validity, not creativity, is the outcome the structural comparisons can be read on.**
- **The ICML 2027 main track is no longer recommended.** This follows the Nov 8 decision rule in
  `plan/icml2027-review-20261017.md` §0 ("Sibling fails → NeurIPS / COLM / TMLR"). The material can still make a
  sound analysis paper (§4).

**Corrections to my own earlier review**
- I predicted that gains would be "monotone, saturating in d_Y". G14 falsified that.
- My factorial table described G22 vs G12 as "target-side multiplicity added on top" without noting that G22 also
  doubles the answers (5k → 10k) and the edges. The amendment's "answer count fixed" error came from that description.

---

## 2. A chance model for novelty

**Structure space.** The evaluator canonicalizes Circle and Line answers by first-occurrence relabelling.
- For an n-node directed cycle or path whose edges are listed in random order, the number of distinct canonical
  structures is **S = (n−1)!**. This is verified exhaustively for n = 3–6 in the test, and by sampling for the real
  tasks: 400k Circle draws saturate at 40,319 of 40,320.
- **Circle** (9 nodes): S = 40,320. Its 10k training answers contain 8,859 distinct structures, a coverage of
  c = 0.220. fresh20k covers 0.390; G12 0.116; G14 0.060.
- **Line** (10 nodes): S = 9! = 362,880, so c ≈ 0.027.
- **Sibling** (derived 5×500): S = Σ_a |b1(a)|·|b2(a)| = 1.25M, so c ≤ 0.04.

**Decomposition.** creativity = V × (1 − seen/V) × u, where V is validity and u the deduplication ratio.
- If valid outputs were uniform draws from the space, then seen/V = c.
- Define the **copy ratio** r = (seen/V) / c. r ≈ 1 means the model reproduces training structures exactly as often
  as chance would.
- Define **C@c0** = V·(1 − c0·r)·u: creativity with every cell moved to a common reference coverage c0 = 0.220.

**Circle.** "hist" rows are stage0b-mult held-out means; "new" rows are from the uploaded report.

| Cell | Src | Edges | C | V | seen/V | chance c | Copy ratio r | C@0.220 |
|---|---|---|---|---|---|---|---|---|
| fixed | hist | 10k | 0.162 | 0.211 | 0.225 | 0.220 | 1.02 | 0.162 |
| b128 | hist | 10k | 0.165 | 0.211 | 0.213 | 0.220 | 0.97 | 0.165 |
| G12 | new | 10k | 0.218 | 0.279 | 0.213 | 0.116 | 1.83 | 0.166 (0.217*) |
| G14 | new | 10k | 0.059 | 0.111 | 0.066 | 0.060 | 1.10 | 0.048 (0.049*) |
| fresh20k | new | 20k | 0.194 | 0.328 | 0.401 | 0.390 | 1.03 | **0.251** (0.253*) |
| 2hash | hist | 20k | 0.241 | 0.319 | 0.240 | 0.220 | 1.09 | **0.241** |
| G22 | new | 20k | 0.246 | 0.370 | 0.324 | 0.220 | 1.48 | 0.246 |
| 2tgt | hist | 20k | 0.294 | 0.407 | 0.270 | 0.220 | 1.23 | 0.294 |
| k2c / k3c / k4c | hist | 20k / 30k / 40k | 0.244 / 0.200 / 0.176 | 0.392 / 0.416 / 0.498 | 0.37 / 0.50 / 0.62 | 0.220 | 1.69 / 2.26 / 2.81 | 0.244 / 0.200 / 0.176 |

For cells already at c = 0.220, C@0.220 is their measured creativity. For cells at another coverage, the main value
keeps the cell's own copy ratio. \* The bracketed value assumes chance-level copying (r = 1) instead. Which one applies
is untested: it depends on whether a cell's copying bias would persist with a larger answer set. G14's low C also
reflects heavy duplication (u = 0.57).

**Line** (`code/coverage.py` on `runs/line-phase1`; coverage from Line data regenerated with the datagen default
seed, since no Line data SHA is recorded): the copy ratio r is

| fixed | b128 | 2tgt_sample | 2hash | 2tgt |
|---|---|---|---|---|
| 0.75 | 1.14 | 1.19 | 2.45 | 3.06 |

At c = 0.027, copying barely moves creativity, so Line's 2hash gain (+0.034) is a validity gain (+0.045).

**What this shows**
1. **One-to-one pairing yields chance-level novelty.** For fixed, b128 and fresh20k on Circle, and fixed and b128 on
   Line, the valid outputs coincide with training structures at exactly the coverage rate. So the "novelty" part of
   creativity on Circle is a property of the training-set size, not of the model. For seed-conditioned models of
   this kind, creativity on Circle ≈ 0.78 × validity. This bears on how the original benchmark's metric should be
   read; check whether its Circle configuration matches ours before stating it.
2. **Multiplicity raises copying.** Copying rises above chance with target-side multiplicity (2tgt, G22,
   k2c → k3c → k4c, and randperm at the limit) and with G12. It is also above chance for 2hash on Line. This
   is the most consistent structural effect across cells and tasks.
3. **Validity on unseen seeds tracks the number of distinct supervised (hash, answer) edges.**

   | Distinct edges | Validity V |
   |---|---|
   | 10k | 0.21 |
   | 20k | 0.32–0.41 |
   | 30k | 0.42 |
   | 40k | 0.50 |

   Exceptions and confounds:
   - G12 (0.28) and G14 (0.11) at 10k edges are exceptions in both directions, so the number of distinct answers
     matters at the low end.
   - The k-cycle cells also train on more sequences per step (128 / 192 / 256).
4. **2hash vs fresh20k:** equal validity (0.319 vs 0.328) and equal coverage-adjusted creativity (0.241 vs 0.251).
   Doubling the edges explains 2hash's Circle gain as well as multiplicity does.
5. **G12 vs b128:** G12 has higher validity (0.279 vs 0.211; historical reference), but its creativity lead comes
   from having half the training answers (c = 0.116). At equal coverage it is 0.166–0.217 against b128's 0.165, so
   the creativity comparison cannot be identified.

---

## 3. How far the factorial comparisons can be trusted

- **Historical references can move substantially.** The same cell, 2hash seed 42, run twice:

  | Execution | V | seen | C |
  |---|---|---|---|
  | stage0b-mult | 0.306 | 0.069 | 0.235 |
  | Oct 17 rerun (regenerated dup2 data, `circle-exposure-dev-rescore`) | 0.428 | 0.208 | 0.220 |

  Creativity agrees within 0.015, but validity and copying differ by 0.12 and 0.14. So comparisons of V or r against
  historical b128 and 2hash are only good to about ±0.1.
- **Answer-set size cannot be held fixed** while varying d_Y at fixed hash count, because N_a = N_h·d_H/d_Y. Holding
  N_a fixed means varying N_h, as in b128 → 2hash. Any structural claim needs at least two cells that differ in only
  one of {N_h, N_a, edges}. The current set has none without a historical reference.
- **Every Circle cell reuses the development held-out file** (probe SHA `7b20…454c`), as the report states.

---

## 4. What the paper can claim now, and the venue

**Supportable with existing data plus `coverage.py`** (no new training):
1. **Measurement.** Creativity on these tasks decomposes into validity × (1 − coverage × copy ratio) × uniqueness.
   Circle's space is 8! and Line's is 9!. With one-to-one pairing, novelty sits exactly at chance.
2. **Copying.** Target-side multiplicity converts validity into copying, monotonically in k (k2c → k4c → randperm).
   This is the clearest structural regularity.
3. **Boundaries.** No pairing scheme helps on all three tasks. Batch size or exposure alone does nothing on any task.
4. **Collapse.** TL + NTP on the same map collapses. This is single-seed: rerun with 3 seeds or keep it as an
   observation.

**Needs contemporaneous controls before it can be claimed:** that validity tracks the number of distinct edges, and
anything about answer-side multiplicity.

**Venue**
- **TMLR** (rolling) fits an analysis and boundary paper with this protocol rigour best.
- **NeurIPS 2027** (≈ May 21, estimate) is the conference target, if §5 items 3–4 support the edge-count reading.
- **COLM 2027** (≈ late March, unconfirmed) is a reasonable middle option.
- **ICML 2027** is not recommended. A January submission would rest on a reframed claim that is only days old.

---

## 5. Next steps (cheap; no new components, no k sweeps, no scale)

| # | Item | Cost | Decides |
|---|---|---|---|
| 1 | Sibling decomposition: V, seen, dedup and memo for the 9 runs; `coverage.py --sibling_graph <accepted graph>` | none | Whether 2hash loses on Sibling through validity or through copying (r) |
| 2 | Run `coverage.py` over all Circle runs on the server (stage0b, stage0b-mult, the factorial) and archive the JSON | none | Replaces the hand-computed table in §2 |
| 3 | Contemporaneous Circle references: graph (10k, 1, 1) at batch 128 and graph (20k, 1, 2), same environment as the factorial, seeds 42–44 | 6 runs, ≈ 10 GPU·h | Whether fresh20k ≈ 2hash, and G12's validity lead over b128, survive within one execution |
| 4 | Line: fresh20k + contemporaneous b128 and 2hash, seeds 42–44 | 9 runs, ≈ 15 GPU·h | Whether Line's 2hash gain is "more edges" |
| 5 | Report validity and copy ratio as co-primary structural outcomes; creativity as the composite | none | Removes the coverage artifact from future verdicts |

Items 1, 2 and 5 cost nothing and should come first. Items 3 and 4 are worth running only if the paper keeps the
edge-count claim.
