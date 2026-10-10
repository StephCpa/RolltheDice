# External review of P6 stage 0: code, evidence, and next steps (2026-10-10)

**Scope.** I read every document in this repository (proposal, Meta-method review, verification log, stage-0 plan, run registers, reports) and all of `code/`. I re-analysed the committed `scores.jsonl` / `churn_analysis.json` files and checked claims by running the code on CPU (a tiny randomly initialised GPT-2 and an offline tokenizer, so no Hub access was needed). I also compared the port line by line with the original repository at `ChenWu98/algorithmic-creativity@7176dc7` (`circle-construction/ntp/{train.sh,eval_qa.py,main.py}` and `simpletransformers/.../seq2seq_model.py::predict`).
**What this review is not.** I ran no GPU training and produced no new results on the real task. Every number quoted below comes from files already committed here, or from the CPU probes described in §4.

---

## 0. Summary, in priority order

| # | Finding | Effect on current conclusions | Status in this branch |
|---|---|---|---|
| **P0-1** | `Evaluator.seen()` does not cut the output at `</a>`. Batched `generate` pads rows that finish early with `<|endoftext|>`, so a memorised circle in such a row is scored **unseen/novel**. | `test_seen` is too low, and `test_unseen` and `n_unique_valid_novel` are too high, for **every circle run**. `test_validity`, `train_memorization` and `n_unique_valid` are not affected. | **Fixed**, with a regression test. Every eval now saves its raw generations. |
| **P0-2** | The reports draw conclusions from runs that the incident register marks INVALID or DEVIATION. | The "G1: non-permanence alone does not help" conclusion has no support. The setc lead was measured on the DEVIATION run. | Warning banners added to both reports. |
| **P0-3** | The setc vs ntp_fixed comparison is confounded by the teacherless objective, 2× forward/backward passes per step, and loss scale. | "setc leads" cannot be credited to *optimal matching*. | **Added** the matched controls `tl_fixed_m4` and `tl_randperm_m4`. |
| **P0-4** | Too little statistical power: 128 test probes and one training seed. | Headline +0.039 validity: z = 0.74, p ≈ 0.46. Two runs that are effectively the same configuration differ by 0.063. | Run plan in §2; `rescore.py` re-scores with up to 2048 probes. |
| **P0-5** | "No-seed mode collapse reproduced" follows from greedy decoding on a constant prompt and holds by construction. | It is not evidence for the paper's claim. | `rescore.py --sample` runs the fair comparison. |
| P1-6 | Freezing is the expected fixed point of hard-assignment training with fixed groups and seeds that have no structure. | Freezing is not an empirical surprise, and the current design cannot escape it. | Added `--regroup`; `--m 64 --groups_per_step 1` already works. |
| P1-7 | Hungarian ties resolve to the identity: 73.5% identity in steps 0–199, against 4.2% under chance. | setc warm-up trains the dataset's nominal pairing. | **Added** `--random_tiebreak`. |
| P1-8 | `train_memorization` is misread for setc. | "setc does not memorise (0.41 vs 1.00)" has no support. | Documented in the evaluator. |
| P1-9 | The plug-in I(z;[y]) proxy reads ≈5 bits even when z and y are **independent**. | The planned measurement would have looked like strong seed use. | **Fixed:** permutation null, plus a likelihood-based `--mode mi`. |
| P1-10 | The C01 speedup (≈m) is forward-pass arithmetic; the open question is whether the cheap cost picks the same pairing as the AR likelihood. | C01 as currently stated is trivially true. | **Added** `diagnose.py --mode agree`. |
| P1-11 | `teacherless_readout` appends a second `</a>`, so validity is always 0. | The planned readout diagnostic would report 0. | **Fixed**, with a test. |
| P2-12 | The closest prior work is missing: minibatch-OT flow matching, Immiscible Diffusion, MCL/WTA collapse. | The novelty boundary is too generous. | §3.3 |
| P2-13 | Circle and Sibling have no constraint *c*, so they cannot test the central hypothesis. | They limit what stage 0 can show, not its correctness. | Suggested a bridging task in §3.4 |
| P2-14 | Process: no semantic unit tests, predictions not saved, reports edited by hand. | Incidents #3 and #4 would have been caught by tests. | `code/test_pipeline.py` (14 tests); `requirements.txt` |

---

## 1. Correctness findings (with evidence)

### 1.1 P0-1 — `seen()` and trailing pad tokens (incident #4)

* `run_eval` decodes with `skip_special_tokens=False`. In a batch of 32, any row that emits `</a>` before the longest row is padded with `pad_token_id = eos`. I confirmed this with a CPU probe: row 0 came back as `[10, 11, END, 0, 0, 0]`.
* `Evaluator.seen()` passed `pred.split("<q>")[1]` straight to `circle_canonicalize`. That function strips a trailing `</a>` only when the string *ends* with one. With pads present, the last edge splits into more than two parts and the function returns `None`, so `seen = 0`. A valid memorised circle is then counted as unseen and novel. This reproduces with the committed evaluator:
  `clean → valid 1, seen 1`; `clean + "<|endoftext|>"×3 → valid 1, seen 0`.
* The original code never had this problem: `seq2seq_model.py::predict` does `outputs[i].split("</a>")[0].strip()+"</a>"` and decodes with `skip_special_tokens=True`. The bug is specific to the port.
* **The bias does not cancel between arms.** It fires only when a valid answer shares a batch with a longer output, usually an invalid over-length one. Arms that produce more over-length outputs therefore have more of their memorised answers relabelled as novel.
  The committed data contains a symptom. ntp_fixed `test_seen` goes 0.086 → **0.000** → 0.0625 at 30k/35k/40k while validity stays flat (0.219/0.219/0.211). At 35k every one of its 28 valid outputs counted as novel.
* **Fix:** `seen()` now truncates at the first `</a>`, and `run_eval` and `diagnose.sample_hashes` truncate as well, which mirrors the original's post-processing. Each eval now writes `preds_step{N}.jsonl`, so later metric fixes need no regeneration.
  **Action:** re-score every existing `final_model` with `code/rescore.py`, including the server runs started before this fix.

### 1.2 P0-2 — Reports still cite invalid runs

`runs/stage0-report.md` §1 and `stage0-progress-summary-en.md` both list `ntp_randperm` as the G1 control and conclude that "non-permanence alone does not help". `ntp_randperm_INVALID-nominal-ntp/INVALID.md` records that this run **never re-paired**. Both reports also present `setc_m4` as "leading on every generative metric", but it is the DEVIATION run.

The invalid run is still useful. Under the bug it trained exactly the dataset pairing, so it is **effectively a replicate of ntp_fixed** that differs only in batch composition and data order. That gives a free estimate of the noise floor:

| run (circle, 40k, 128 test probes) | validity @40k | mean over 20k–40k checkpoints | range over 20k–40k |
|---|---|---|---|
| ntp_fixed | 0.211 | 0.203 | 0.180–0.219 |
| ntp_fixed replicate (randperm-INVALID) | 0.148 | 0.192 | 0.148–0.234 |
| setc_m4 (DEVIATION) | 0.250 | **0.255** | 0.227–0.289 |

* The two fixed-pairing runs differ by **0.063** at 40k. Within one run, validity swings by **0.086** between neighbouring checkpoints (replicate, 35k → 40k). The headline setc lead (+0.039 validity, +5 unique) sits inside both.
* One pattern is worth keeping: **setc is at or above 0.227 at every late checkpoint**, and its late mean is about 0.05 higher than both fixed runs. That is suggestive, but the checkpoints share the same 128 probes, so they are strongly correlated, and the comparison is confounded (§2.1). Treat it as a hypothesis for the matched, multi-seed design.

I added a short warning banner to both reports and left their content unchanged.

### 1.3 P0-5 — The no-seed arm under greedy decoding

Every no-seed probe is the identical prompt `<q>` (`circle_form_test("")`), so greedy decoding *must* return a single output. "1 unique valid, novelty 0" is therefore guaranteed by the evaluation protocol and says nothing about the model. The original repository's in-training `predict` is also greedy (`do_sample=False`), so the port matches it. The paper's claim, however, is about seed-conditioning *versus temperature sampling*: your own `verification/2026-10-09-web-checks.md` cites the §5.1 discussion. A reproduction needs:
`python code/rescore.py --model_dir <noseed>/final_model --data_dir data/circle.10.9.0.10000 --out <noseed>/rescore_T1 --n_probes 2048 --sample --temp 1.0`
(and T ∈ {0.5, 1.0, 1.5}), compared with the seed arms at the **same number of generations**.

### 1.4 P1-7 — Hungarian tie-breaking resolves to the identity

`linear_sum_assignment` returns the identity for an exactly tied matrix: 100% of trials for a constant matrix, and still 41.5% after fp16 rounding with a 0.01 spread. Ties are common early in training, for three reasons:

1. When the model ignores z, `C[i,j] = c_j` is constant down each column, so **every permutation has the same total cost** Σ_j c_j.
2. All circles share nearly the same token multiset.
3. The logits are fp16.

The committed `churn_analysis.json` shows identity in **73.5%** of assignments during steps 0–199 (n = 3188), against 1/24 = 4.2% by chance. Setc warm-up therefore trains the nominal dataset pairing, exactly what the method is meant to move away from. In the frozen phase (steps ≥ 12k) identity stays at about twice the chance rate: 7.2–8.7% per 200-step window, mean 7.8%.

**Fix:** `--random_tiebreak` randomly permutes the columns before solving and maps the result back, using its own RNG stream. It is off by default so that in-flight runs remain reproducible; I recommend turning it on for every future setc run.

Related detail: the costs are computed in train mode, with dropout 0.1 active. During the "symmetric" phase the assignment is therefore largely driven by dropout noise. That is not a bug, but it should be stated whenever phase 1 is interpreted.

### 1.5 P1-8 — What `train_memorization` measures for setc

`train_memorization` is an exact match against the **dataset's** hash→target pair. After freezing, setc reproduces *its own* pairing y_π\*(i). If that pairing is a uniformly random permutation, the expected score is 0.25 (the expected share of fixed points), even when memorisation is perfect.
The observed 0.406 is above that, which fits the identity bias (§1.4, plus the incident-#3 NTP term). It does **not** show an absence of memorisation. To report memorisation for setc, use:
(a) P(output ∈ training targets | training hash), i.e. `seen` computed on training probes; and
(b) P(output = y_π\*(i)), using the final permutations in the churn dump.
Both can now be computed from `preds_step*.jsonl`.

### 1.6 P1-9 — The I(z;[y]) proxy needs a null

The plug-in estimator is capped by sample size: H([y]|z) ≤ log₂16 = 4 bits and H([y]) ≤ log₂512 = 9 bits. On circle, almost every sample forms its own class (the training data already contains 8,859 distinct structures in 10,000), so **a model that ignores z still scores ≈5 bits**. `test_plugin_I_proxy_is_explained_by_null` demonstrates this with synthetic, z-independent samples (I ≈ 5 bits, null ≈ 5 bits).
**Fixes:** (i) `--mode seeds` now also reports a permutation null (samples reassigned to random hashes) and `I_proxy_minus_null_bits`, and counts **valid** outputs only. The old coverage function was counting any parseable string. (ii) The new `--mode mi` estimates Î(z;y) = E[log p(y|z_k) − log (1/K)Σ_k' p(y|z_k')] from the model's own likelihoods. This is exact for z uniform over the K probe seeds, bounded by log₂K, and has no class-counting bias. A unit test checks that it returns exactly 0 when all prompts are identical.

### 1.7 P1-11 — `teacherless_readout` double `</a>`

`L_ans` already includes the `</a>` slot, and the code appended another, so `validity()` (which requires exactly one `</a>`) always returned 0. This is fixed and tested with an oracle model.

---

## 2. Experimental design

### 2.1 P0-3 — Confounds and matched controls

| arm | objective | fwd+bwd passes / step | examples / step | pairing rule |
|---|---|---|---|---|
| ntp_fixed | NTP | 1 × 64 | 64 | fixed (dataset) |
| ntp_randperm | NTP | 1 × 64 | 64 | random bijection per visit |
| tl_hybrid | 1 NTP : 3 MTP inside 64 units | 1 × 64 | 16 | fixed |
| setc_m4 | NTP + 3·MTP (loss summed: about 4× hybrid's scale) | **2 × 64** | 64 | Hungarian; **frozen after ~12k steps** |
| **tl_fixed_m4** (new) | same as setc | 2 × 64 | 64 | fixed (identity within group) |
| **tl_randperm_m4** (new) | same as setc | 2 × 64 | 64 | random bijection per visit |

After about 12k steps (70% of training), setc *is* "teacherless + NTP with a fixed pairing it chose itself". Only `tl_fixed_m4` and `tl_randperm_m4` isolate the selection rule: they share setc's objective, loss weight, compute, batching and code path, and differ only in how π is chosen.
Both arms also log the Hungarian result as an `argmin` field without using it. For tl_fixed, `argmin == perm` measures how far the model's preferences have come to match the imposed pairing, and `diagnose.py --mode churn` reports this as `argmin_agree_frac`.
**I verified that existing arms are unchanged.** The edited trainer reproduces the original's setc churn log, cost-matrix dumps, and the final weights of `setc_m4`, `ntp_randperm` and `ntp_fixed` **bit-for-bit** (tiny model, 20 steps).

Two further notes:
* **Compute matching.** setc and the tl_* arms cost about 2× ntp_fixed per step. If you ever compare with NTP arms, include ntp_fixed at 80k steps or batch size 128.
* **`--mtp_weight`** is now a flag. The default of 3.0 is unchanged.

### 2.2 P0-4 — Statistics

* With p ≈ 0.23 and n = 128, the binomial SE is 0.036 *for evaluation noise alone*. Detecting +0.04 at α = 0.05 with power 0.8 needs about 1,700 test probes per arm (unpaired). `test.json` already holds 1,024 test and 1,024 train probes, of which training uses only 128 + 128. Pass `--n_probes 2048`; for the confirmatory run, generate ≥4,096 fresh unseen hashes, which cost nothing to make.
* Training-seed variance is at least as large as evaluation variance (the replicate gap of 0.063). Use **at least 3 seeds per decisive arm**. The unit of analysis is the training seed; within a seed, compare arms on the *same* probes (paired).
* **Pre-register the endpoint** so you never pick the best checkpoint. For example: the mean over the last three checkpoints (30k/35k/40k) of validity, unseen-valid rate, and unique-novel per probe on the fixed probe set; plus U_N at T = 1 with a fixed budget N.

### 2.3 Status against the stage-0 gate (plan §0)

1. *Direction-consistency*: not yet testable. It needs the no-seed arm under temperature sampling (§1.3) and the Sibling task, which has not been run; the gate requires both tasks.
2. *Evaluation verification*: the self-test missed P0-1. Now that predictions are saved, add a **differential test**: apply the original's `split("</a>")[0]+"</a>"` post-processing, run the original `eval_qa.Evaluator.eval_items` on the same `preds_step*.jsonl`, and compare its output with the port's on every field.
3. *C01*: the microbenchmark is pending. See §3.2 for what it should measure.
4. *Diagnostics*: churn works. The I proxy needed the fixes in §1.6.

### 2.4 Suggested run plan (8× A6000 server)

| phase | runs | cost |
|---|---|---|
| **A. Re-score** (no training) | `rescore.py` on every existing `final_model` with 2048 probes; the no-seed arm with `--sample` at T ∈ {0.5, 1, 1.5}; `diagnose.py --mode readout/mi/agree` on setc, tl_hybrid_v2 and ntp_fixed | minutes per model |
| **B. Matched comparison** | {ntp_fixed, tl_fixed_m4, tl_randperm_m4, setc_m4 `--random_tiebreak`} × 3 seeds × 40k, all with `--n_probes 2048`; tl_hybrid × 3 seeds if budget allows (`bash code/run_stage0.sh circle 40000 <seed>` with `EXTRA="--n_probes 2048 --random_tiebreak"`) | On your 2080 Ti timings: about 1.6 h per ntp/hybrid run and about 4.4 h per setc/tl run (0.39 s/step p50), so about 49 GPU-h for three seeds. That uses the rest of the 60 h stage-0 cap (about 10.5 h already spent). A6000 time is unmeasured and probably lower. |
| **C. Escaping the freeze** (stage-1 candidates, 1 seed first) | setc `--regroup`; setc `--m 64 --groups_per_step 1` (whole-batch matching at **no extra forward cost**, since the cost matrix is a gather); setc with Sinkhorn or annealed assignment | about 4.4 h each |

The decision rule follows plan §2. Read B as follows. If setc beats tl_fixed_m4, the self-selected pairing helps. If setc ≈ tl_fixed_m4 > ntp_fixed, the gain comes from the teacherless objective, which is the original paper's result rather than ours. If tl_randperm_m4 ≈ setc, non-permanence explains the gain.

---

## 3. Mechanism, theory and positioning

### 3.1 P1-6 — Freezing is the expected outcome here

* **Phase 1 is Hungarian on noise.** While the model ignores z, every permutation ties in total cost (§1.4). The assignment then depends only on the *interaction* term, C minus its row and column means: tiny seed effects plus dropout. Churn ≈ 1.917 is exactly the uniform-S₄ value, 4 − H₄.
* **Phase 2 onward is hard EM.** The Hungarian step is a hard E-step and the gradient on the matched pairs is the M-step. As soon as any seed dependence appears, the step reinforces it, with the same rich-get-richer dynamics as k-means and as winner-takes-all in multiple-choice learning, where it is known as *hypothesis collapse* and sensitivity to initialisation.
* With **fixed groups of four** and **hash seeds with no geometry**, each group has only 24 fixed points, and memorising any of them is self-consistent. Freezing is the generic outcome, so it should not be presented as a surprising finding.
* **Cheap diagnostic:** the matrix dumps saved every 50 steps already let you plot ‖interaction‖ and the gap between the best and second-best permutation over training. Phase 2 should begin where the gap exceeds dropout-level noise. This turns the three-phase story into a measured quantity.

**What could make set-coupling differ from a fixed pairing?**
1. **Change the candidate set** (`--regroup`, or matching over the whole batch with `--m 64`). There is then no fixed permutation to freeze into, and the pairing has to generalise. This is the real "distribution-level coupling" that the proposal describes.
2. **Seeds with geometry.** OT couplings help flow matching because they make a map *in a metric space* simpler (straighter). Random 10-character hashes have no metric, so the "optimal" coupling can only exploit what the model has already memorised. A stage-1 variable worth testing: structured seeds (overlapping seed tokens, or continuous seed embeddings) for which a learnable seed → solution map can exist.
3. **Annealing or stochastic assignment:** Sinkhorn with a decreasing temperature (already planned), Gumbel-perturbed Hungarian, or annealed MCL.

### 3.2 P1-10 — What C01 should claim

Computing m² teacherless costs from m forward passes is true by construction, and I verified the vectorised gather against brute force (max error 1.9e-6). The speedup over a naive m² implementation is therefore about m by counting forward passes, which is not an empirical finding. The substantive question: **does the teacherless cost select the same pairing as the AR likelihood that the model uses at generation time?** `diagnose.py --mode agree --train_seed <seed>` rebuilds the training groups and reports four things on a trained checkpoint:
* the exact agreement of π\*_TL and π\*_AR;
* the Cayley distance between them, against chance;
* the AR-likelihood regret of using π\*_TL;
* the Spearman correlation between the two cost matrices.

Low agreement would mean the cheap matching optimises a different objective from the one that matters.
On `throughput.py`: it copies full `[64,128,V]` fp32 log-probabilities to the host (about 1.6 GB per chunk), so its timings mostly measure host transfer. It also uses random tokens on a pretrained model, so its cost-divergence number is uninformative. Gather on the GPU and use a trained checkpoint on real data.

### 3.3 P2-12 — Related work the novelty boundary must cover

On top of IMLE and OTSeq2Seq/OTSeq2Set, which are already listed:
* **Minibatch OT couplings in flow matching.** Pooladian et al., *Multisample Flow Matching*, ICML 2023 (arXiv:2304.14772). Tong et al., *Improving and generalizing flow-based generative models with minibatch optimal transport*, TMLR 2024 (arXiv:2302.00482). Both pair noise samples with data samples inside a minibatch using an exact or entropic OT assignment, which is structurally the closest analogue of "seed set ↔ solution set".
* **Immiscible Diffusion**, Li et al., NeurIPS 2024 (arXiv:2406.12303). It assigns noise to images by **linear assignment within a minibatch** to speed up diffusion training: the same Hungarian noise↔data pairing, in a continuous domain.
* **Multiple-choice learning / winner-takes-all.** Guzman-Rivera et al., NeurIPS 2012; Lee et al., *Stochastic MCL*, NeurIPS 2016; Perera et al., *Annealed Multiple Choice Learning*, NeurIPS 2024 (arXiv:2407.15580). This line documents hypothesis collapse (≈ your freeze) and annealing as its remedy.
* **Hungarian-matched set losses.** DETR (Carion et al., ECCV 2020) and permutation-invariant training for speech separation (Yu et al., ICASSP 2017).

Against these, the defensible differences are narrower: discrete sequences, **a cost from teacherless parallel prediction** (m forward passes for m² costs), verified solution sets, and evaluation on unseen seeds and constraints. Section 2 of the Meta-method review (`analysis/2026-10-09-meta-method-review.md`) should be updated accordingly.

### 3.4 P2-13 — A bridging task that can test the hypothesis

Circle and Sibling have no constraint *c*: every training answer is valid for every seed, and a "group" is just a random subset. Set-coupling over Y_c and compositional generalisation cannot be tested on them, and coverage cannot be measured.
A cheap bridge that reuses the circle infrastructure: **c = a node subset S with |S| = k given in the prompt, and Y(c) = the Hamiltonian cycles on S.** There are |Y(c)| = (k−1)!/2 distinct undirected cycles (60 for k = 6, 360 for k = 7). This makes coverage **exactly enumerable**, puts m solutions *of the same c* into each group as the proposal intends, and lets you hold out subsets (or combinations of subset size and node) for generalisation tests.

---

## 4. Changes in this branch (`claude/zen-bohr-gq1si2`)

| file | change |
|---|---|
| `code/evaluator.py` | `seen()` truncates at `</a>` (P0-1); 2 new self-test cases (15/15); docstring notes the setc memorisation semantics and the metric differences from the original. |
| `code/train.py` | Truncates answers at `</a>` in `run_eval`; saves `preds_step{N}.jsonl` (opt out with `--no_save_preds`). New arms `tl_fixed_m4` and `tl_randperm_m4`. New flags `--random_tiebreak`, `--regroup` (churn records then carry `members`), `--mtp_weight`. Removes the duplicate final eval row. **Existing arms are bit-identical in training** (verified). |
| `code/diagnose.py` | Validity-filtered, truncated class keys; permutation null for the I proxy; double-`</a>` readout fix; new `--mode mi` and `--mode agree`; churn analysis infers m, handles `--regroup`, and reports `argmin_agree_frac`. |
| `code/rescore.py` (new) | Re-scores a saved `final_model` (more probes, greedy or sampled), reusing `run_eval`. |
| `code/test_pipeline.py` (new) | 14 offline CPU tests (`python code/test_pipeline.py`). Covers pairing semantics (incident #3), cost matrix vs brute force, no leakage, tie-breaking, eval/readout truncation (incident #4), the I-proxy null, MI and `seq_logprobs`. Mutation-checked: reinstating either bug makes the relevant tests fail. |
| `code/run_stage0.sh` | `PY` and `EXTRA` overrides (the Windows venv path is no longer hard-coded); seed argument; the two matched-control arms. |
| `requirements.txt` (new) | Pins the server versions recorded in `runs/README.md`. |
| `runs/stage0-report.md`, `runs/stage0-progress-summary-en.md`, `runs/README.md`, `README.md` | Warning banners or incident-#4 entries only. Original text is untouched. |

**Not done, needs your call or a GPU:** re-scoring the existing models; the matched multi-seed runs; Sibling; the differential test against the original `eval_qa.py`; committing the raw 40k churn dump (README says the churn records are kept as gzip, but no `.gz` is in the repository, and the freeze finding depends on that data); rewriting the report tables (best regenerated from `scores.jsonl` by a script rather than by hand, so they cannot drift from the incident register again).
