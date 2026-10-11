# P6 → ICML 2027: feasibility assessment and experimental plan

Draft for the project lead, 2026-10-11. Written against `main` @ `f6bd12a`.
Deadline facts: the ICML 2027 call for papers was not yet published when this was written. A deadline tracker
estimates abstract ≈ **Jan 16, 2027** and full paper ≈ **Jan 22, 2027** (AoE), and puts the venue in South America.
For calibration, ICML 2026 had abstract Jan 23 and paper Jan 28, 2026. **Plan to an internal freeze of Jan 15, 2027**
and re-check the official dates as soon as the call is out.

---

## 0. Bottom line

**ICML 2027 is ambitious but feasible — as an empirical "understanding" paper, not as a set-coupling method paper,
and only if two conditions hold:**

1. **The effect generalizes beyond Circle.** It must reproduce on at least one more task of the original benchmark (Line is
   cheap; Sibling is the decisive "discovery" task), and keep its sign at one larger model scale.
2. **Compute access.** The full plan needs ≈ 900–1,200 A6000·h over ~8 weeks, i.e. **≥ 4 dedicated A6000s**. On the current
   single shared GPU it is not feasible.

The paper's core is what the project has actually found, not what it set out to build:

> In seed-conditioned training (Nagarajan et al., ICML 2025), adding the teacherless (parallel-prediction) objective on the
> *same* seed→answer map turns seeds into lookup keys. The model memorizes the map exactly and loses the ability to
> produce valid novel structures from unseen seeds. **Giving the teacherless branch a *different* target than the
> next-token branch prevents this collapse and raises creativity above the next-token baseline.**

This speaks directly to the original paper's open question (its §B.1): why arbitrary seed–answer pairings generalize. It also
explains its negative small-model teacherless result. The Hungarian set-coupling idea becomes one arm of the story
(optimal coupling freezes into a self-chosen pairing), not the contribution.

Two decision points determine whether January stays realistic: **Gate 1 on Nov 8** (generalization) and **Gate 2 on Dec 6** (scale).
If either fails, the recommended fallback is **NeurIPS 2027** (deadline ≈ mid-May, which allows a broader study) or **TMLR**
(rolling) for the careful small-scale version. §8 has the details.

---

## 1. Where the project stands (progress check)

| Item | Status | Strength |
|---|---|---|
| Primary contrast (pre-registered): `setc_MN` vs `ntp_fixed`, Circle, held-out 2,048 hashes | **PASS**: 0.209 ± 0.006 vs 0.162 ± 0.009; per-seed Δ +0.041 / +0.050 / +0.050 | Solid for one task / one model / one setting; seed 42 was used during design (flagged) |
| Collapse cells (TL & NTP on the same map): `tl_NN`, `setc_NM`, `setc_MM`, `tl_hybrid_v2` | All lose validity after memorizing (memo_own ≈ 1, nll_own ~1e-5) | 1 seed each (hybrid: 2 environments); needs replication |
| Gain decomposition C = V × Q | Gain tracks validity; conversion Q unchanged (0.768 vs 0.766) | Descriptive |
| Late intervention (`setc_MM` → NTP nominal at 20k) | 0.023 → 0.227 (dev endpoint); held-out 0.237 at selected ckpt | 1 seed; 20k is near the MM peak → shows **prevention**, not yet reversal |
| Matching dynamics (freeze, rich-get-richer, gap 0.25 → 479 nats, commitment ~11k) | Reproduced in two versions | Descriptive, solid |
| Lock check | 99.67% agreement at 14k; divergence expected from ~0.3% residual flips | Re-check from a fully frozen state still open |
| Conflict controls | **`tl_RN` / `ntp_2tgt` were run with the wrong partner spec (incident #7)** | See below |
| Engineering & process | 30 offline tests, bit-identity checks, pre-registration, 7 logged incidents | Strong; this is a selling point in the paper's appendix |

**Incident #7 (found in this check).** Amendment 1 §2 specifies `tl_RN`'s partner R as a *derangement* (no seed's TL target
equals its nominal target), drawn with seed 777. The runs used `--fixed_pairing random`, which draws a uniform bijection from
seed+2. Reconstructing R exactly: **24.3% of seeds had R = nominal** (0.243 / 0.243 / 0.242 for seeds 42/43/44).
- So `tl_RN/s42` (dev peak 0.242) and the queued `ntp_2tgt` runs are the "uniform" variant. Relabel them `*_unif` and
  keep them: their 24% no-conflict share happens to match `setc_MN`'s π* fixed-point share (25–32%).
- The signed-spec version is now `--fixed_pairing derangement --fixed_seed 777`. `run_stage0b.sh conflict` uses it, and the
  old spec is available as `conflict_unif`.
- **Action now:** stop the queued seed-43/44 uniform runs, then relaunch `conflict` (3 seeds, ≈ 11 GPU·h).

---

## 2. What the paper would claim

Working title: *"Seeds as Lookup Keys: How Teacherless Training Breaks Seed-Conditioned Creativity, and Decoupled
Targets Repair It"* (final wording depends on the necessity results in §4.3).

| Claim | Statement (to be earned) | Current evidence | Needed for ICML |
|---|---|---|---|
| **C1 Collapse** | TL + NTP on one seed→answer map ⇒ exact lookup, loss of valid outputs on unseen seeds | 4 arms × 1 seed, Circle | 3 seeds; ≥ 3 tasks; 1 larger model |
| **C2 Remedy** | Decoupling the TL target from the NTP target beats the NTP baseline on held-out creativity | PASS on Circle (3 seeds) | Replicate on Line + Sibling (+ Triangle stretch); 5 seeds on headline; scale point |
| **C3 Necessity** | Which ingredients matter: online matching? the TL channel? how much conflict? partner persistence? | `tl_RN_unif` s42 dev only | Signed-spec controls; dose–response; persistence; channel |
| **C4 Mechanism** | Gain operates through validity; conflict prevents a shared lookup used by AR decoding; intervention evidence | V×Q, 1 intervention | Reversal test, AR-vs-TL agreement indicator, likelihood MI, probes |
| Appendix | Optimal set coupling freezes (hard-EM dynamics); C01 compute; pre-registration & incident register | Done | Write-up only |

**Not claimed:** "creativity" in general; superiority of Hungarian matching (likely unnecessary); natural-language tasks;
behaviour beyond the tested scales.

---

## 3. Feasibility assessment

**Strengths**
- The finding is clean and counter-intuitive: only one cell of the 2×2 survives, and an intervention reverses the outcome
  within a single trajectory.
- It is tied to a very visible paper and to that paper's own open question.
- The evaluation protocol is unusually rigorous for this area: pre-registered held-out evaluation, dev-only checkpoint
  selection, and an incident register.

**Gaps a reviewer will hit, and the fix for each**

| Reviewer objection | Fix in this plan |
|---|---|
| One toy task, GPT-2 small | §4.2 tasks; §4.4 scale |
| Effect is small (+0.047 abs) | 5 seeds + CIs, same protocol on every task; report relative and absolute gains; coverage curves |
| "Any regularizer would do this" | §4.3 A5: weight decay / label smoothing / dropout baselines on `ntp_fixed` |
| "It's just two targets per seed" | `ntp_2tgt` (no TL channel) + dose–response |
| "Teacherless baseline is mis-implemented" | Line-by-line fidelity to the original repo already verified (data, label masks, hyperparameters); report original `tl_hybrid` too |
| Greedy-only evaluation | Temperature sweeps and sampling-budget coverage curves for every arm (no new training) |
| Novelty vs multi-target / label-noise literature | Positioning in §2; regularizer baselines; related-work audit in December |

**Main risks**

| Risk | Fallback |
|---|---|
| Effect does not replicate on other tasks | Gate 1 → pivot to a mechanism/negative-result paper (TMLR, or an ICML workshop) |
| `ntp_2tgt` ≈ MN | Reframe as "multiplicity of seed targets"; story is simpler but still answers B.1; regularizer baselines become essential |
| Effect vanishes at scale | Report honestly; weaker ICML case; consider NeurIPS 2027 with more scale |
| Compute access | Without ≥ 4 GPUs, cut Triangle and the ~2B model, or move to NeurIPS 2027 |

---

## 4. Experimental plan

All arms use the existing protocol unless stated:
- Primary metric: creativity (unique novel valid per generation).
- Checkpoint selection on dev only, then a one-shot score on a fresh held-out set (2,048 hashes per task).
- Seeds paired by training seed.
- Verdicts from `report.py`, with a pre-registered margin per task.

Each phase starts with a signed pre-registration addendum, written *before* reading new held-out scores.

### 4.1 Phase 0 close-out (Oct 12–18)
- Relaunch the conflict controls with the signed spec (`tl_RN`, `ntp_2tgt` derangement / seed 777, seeds 42–44); score held-out; report.
- **Lock re-check** from `state_step20000` (fully frozen), 2k steps per branch: expect bit-identical weights.
- **Reversal test:** continue `setc_MM` 20k → 30k with `--state_at 30000`, then branch with `--ntp_lambda 1`. Does a
  *collapsed* model recover?
- **Recipe choice for Phase 1 (pre-registered rule):** the simplest arm within the margin of `setc_MN` on held-out becomes
  the "recipe", in this order of preference:
  1. `ntp_2tgt`;
  2. `tl_RN` (fixed derangement);
  3. `setc_MN`.

  All three are still run in Phase 1, because necessity must replicate across tasks.

### 4.2 Phase 1: generality across the original benchmark (Oct 19 – Nov 8)

| Task | Steps (original) | Arms | Seeds | GPU·h |
|---|---|---|---|---|
| Line construction | 40k (like Circle) | noseed, ntp_fixed, tl_hybrid, ntp_randperm, tl_NN, setc_MN, tl_RN, ntp_2tgt | 3 | ≈ 36 |
| Sibling discovery | 800k NTP / 1.2M TL (group TL arms run 800k at 64 examples/step) | ntp_fixed, tl_NN, recipe, ntp_2tgt (+ setc_MN if recipe ≠ MN) | 3 | ≈ 350–460 (≈ 19 h per NTP run, ≈ 36 h per TL run) |
| Triangle discovery (stretch) | 800k / 1.2M | ntp_fixed, recipe | 3 | ≈ 150 |
| Circle (top-up) | 40k | ntp_fixed, recipe: seeds 45, 46 (5 seeds total) | +2 | ≈ 7 |

- No-seed arm: report greedy and temperature T ∈ {0.5, 1.0} at a matched generation budget (rescoring only).
- **Before any Sibling run:** our `datagen` generates `sibling.10.1000.10.50000`, but the original run script uses
  `sibling.5.500.10.50000`. Reconcile against the original notebook and use the paper's configuration.
- Line and Triangle generators and evaluators must be ported (§6).

**Gate 1 (Nov 8, pre-registered):** the recipe passes the same criterion (mean held-out Δ ≥ margin, all seeds positive) on
**Line and Sibling**, and the collapse cell (`tl_NN`) underperforms `ntp_fixed` on both. One-task failure → proceed with an
honest scope statement. Both fail → fallback.

### 4.3 Phase 2: necessity and mechanism ablations (Oct 26 – Nov 29, on Circle + Line; cheap 40k-step tasks)

| ID | Ablation | Levels | Question | Runs | GPU·h |
|---|---|---|---|---|---|
| A1 | **Conflict dose** (share of seeds whose TL target ≠ NTP target, fixed partners) | p ∈ {0, 0.25, 0.5, 0.75, 1} | Is the gain monotone in conflict? Where is the threshold? | 5×2×3 | 60 |
| A2 | TL weight | w ∈ {0.5, 1, 3, 6} | Is it a regularization strength? Does it saturate or hurt? | 4×2×3 | 48 |
| A3 | Partner persistence | fixed derangement vs per-visit random partner (`tl_randperm_m4 --ntp_lambda 1`) vs π*-replay partner | Must the TL partner be persistent? Does a *selected* partner add anything? | 2×2×3 new | 24 |
| A4 | Group size / partner pool | m ∈ {2, 8} | Does a larger pool change anything? | 2×2×3 | 24 |
| A5 | **Regularizer baselines** on `ntp_fixed` | weight decay 0.01 / 0.1, label smoothing 0.1, dropout 0.2 | Is any standard regularizer as good? | 4×2×3 | 24 |
| A6 | Data scale | 2.5k / 40k training targets | Does the gain grow or shrink with data? | 2×2×2×3 | 36 |
| A7 | Interventions | NTP→nominal at {10k, 20k, 30k}; reverse MN→MM at 20k | Prevention vs reversal; is removing conflict sufficient to cause collapse? | 4×3 (Circle) | 12 |

### 4.4 Phase 3: scale (Nov 16 – Dec 6)
- **GPT-2 medium (355M)**: Circle + Line × {ntp_fixed, tl_NN, recipe, ntp_2tgt} × 3 seeds, ≈ 110 GPU·h.
- **~2B model** (Gemma-2B as in the paper if full fine-tuning fits in 48 GB with 8-bit Adam; otherwise Qwen2.5-1.5B):
  Circle + Line × {ntp_fixed, recipe} × 3 seeds, ≈ 120 GPU·h.

**Gate 2 (Dec 6):** the recipe's sign holds on GPT-2 medium for both tasks, and on at least one task at ~2B.
Failure → scope the paper to small models explicitly, or move to NeurIPS 2027.

### 4.5 Analyses (mostly no new training; run as data lands)
1. **Metric panel per arm and task:** creativity, V × Q decomposition, validity, memo_own (vs each branch's own target),
   train_seen, test_seen; temperature sweeps; sampling-budget coverage curves.
2. **Lookup-circuit indicator:** on training hashes, how often the AR output equals the TL parallel readout. Prediction:
   ≈ 1 in collapse cells (a shared lookup); low in decoupled arms (AR follows NTP's target, TL holds the partner).
   `diagnose.py readout` + `rescore.py` already provide both.
3. **Seed dependence on unseen hashes:** likelihood-based I(z;y) (`diagnose.py --mode mi`) per arm across checkpoints,
   to show whether collapse means "lookup on garbage keys" or "seed ignored".
4. **Dynamics:** conflict persistence (fixed-point trajectory), nll_own vs creativity, matching freeze and commitment
   (appendix), TL-vs-AR decision agreement per checkpoint (`diagnose.py --mode gap`).
5. **Representation probes:** linear probes from hash-token hidden states to the nominal and partner targets, per layer
   and per arm, to measure lookup encoding strength. Optionally, attention mass on hash tokens during AR decoding of
   valid vs invalid outputs.
6. **Statistics:** 5 seeds for headline arms (3 elsewhere); paired bootstrap CIs over seeds and probes; Holm correction
   across arm-vs-reference tests; every result reported per seed in the appendix.

### 4.6 Baselines (and why each is there)

| Baseline | Role |
|---|---|
| No seed + temperature sampling | The original paper's reference for creativity |
| `ntp_fixed` (seed-conditioning) | Primary reference; the paper's main recipe |
| `tl_hybrid` (paper's teacherless, 1:3) | Shows the collapse in the paper's own configuration |
| `ntp_randperm` | Inconsistent pairing → copying (validity without novelty) |
| `ntp_2tgt` | Simplest multi-target alternative |
| Regularizer baselines (A5) | Rules out the "any regularizer" explanation |
| SEDD (diffusion), optional | The paper's third family; port only if time allows, otherwise cite the paper's numbers |

---

## 5. Compute budget (A6000·h)

| Phase | Must-have | Stretch |
|---|---|---|
| 0 close-out | ≈ 15 | — |
| 1 generality | ≈ 400–500 (Line, Sibling, Circle top-up) | +150 (Triangle) |
| 2 ablations | ≈ 230 | — |
| 3 scale | ≈ 110 (GPT-2 medium) | +120 (~2B) |
| Evaluation, diagnostics, reruns (≈ 10%) | ≈ 80 | ≈ 30 |
| **Total** | **≈ 890** | **≈ 1,190** |

At 4 dedicated A6000s, the must-have tier is ≈ 9–10 days of continuous compute; Sibling dominates (single runs take up to 36 h, so launch it first). That fits comfortably between Oct 19 and
Dec 6 with headroom for reruns. This needs a new budget approval, since the Phase 0 cap was 60 h.

---

## 6. Engineering prerequisites (week 1–2, parallel with Phase 0 runs)

| ID | Work | Effort |
|---|---|---|
| E1 | Port Line + Triangle generators and evaluators from the original notebooks; self-tests; reconcile the Sibling config (5/500 vs 10/1000) | 3–4 days |
| E2 | Generalize the trainer to `AutoModelForCausalLM` (GPT-2 medium, Gemma/Qwen): tokenizer and added tokens, dummy token, bf16, gradient checkpointing, 8-bit Adam | 4–5 days |
| E3 | Flags: `--fixed_pairing mix:<p>` (conflict dose), `--weight_decay`, `--label_smoothing`, `--dropout` | 1 day |
| E4 | Manifest-driven multi-GPU scheduler: completion = manifest verified (lesson from incident #6), automatic held-out + report, `P6_CODE_SHA` exported on the server | 1–2 days |
| E5 | Analysis scripts: AR-vs-TL agreement, temperature / coverage sweeps, probes, figure generation from `report.json` | 3 days |

Every new flag gets a regression test and a bit-identity check against existing presets, as so far.

---

## 7. Timeline (weeks start Monday)

| Week | Dates | Compute | People / writing | Milestone |
|---|---|---|---|---|
| 1 | Oct 12–18 | Phase 0 close-out (signed conflict controls, lock re-check, reversal) | E1, E3 start; Phase-1 pre-registration | **G0 (Oct 18):** recipe chosen; Phase 0 archived |
| 2 | Oct 19–25 | Line full arm set; Sibling launches (long runs) | E1 done; E2 start | |
| 3 | Oct 26–Nov 1 | Sibling continues; A1 dose + A5 regularizers (Circle/Line) | E2; E5 start | Line verdict |
| 4 | Nov 2–8 | Sibling completes; A2, A3 | E2 done; Phase-3 pre-registration | **G1 (Nov 8):** generality |
| 5 | Nov 9–15 | A4, A6, A7; Triangle (stretch) launches | Analyses 4.5.1–3 | |
| 6 | Nov 16–22 | GPT-2 medium runs | Figure prototypes | |
| 7 | Nov 23–29 | ~2B runs; Triangle completes | Probes (4.5.5) | |
| 8 | Nov 30–Dec 6 | Reruns / top-ups only | Results section draft | **G2 (Dec 6):** scale; experiments frozen except fixes |
| 9 | Dec 7–13 | — | Full draft v0; related-work audit (new 2026 papers) | |
| 10 | Dec 14–20 | Small fixes requested by internal review | Draft v1; internal review | |
| 11–12 | Dec 21–Jan 3 | Buffer (holidays) | Light editing | |
| 13 | Jan 4–10 | — | Draft v2; external reader; reproducibility package | **G3 (Jan 10):** figures frozen |
| 14 | Jan 11–17 | — | Final polish; abstract registration (≈ Jan 16, confirm) | Internal freeze Jan 15 |
| 15 | Jan 18–22 | — | Submit (≈ Jan 22, confirm) | |

**Decision rule at G1 / G2:** if a gate fails, decide within 48 hours between (a) submitting a narrower paper with explicit
scope, (b) moving to NeurIPS 2027 (≈ mid-May) with the broader study, or (c) TMLR. For a project this size, (b) is the
default if G1 fails on Sibling but holds on Line.

---

## 8. Paper outline (figures to aim for)

1. Introduction: seed-conditioning, teacherless training, the B.1 open question; one-figure summary (the 2×2 and the intervention).
2. Setup: tasks, metrics, evaluation protocol (dev selection → held-out; pre-registration).
3. **Collapse:** trajectories of validity / creativity / memo for the four cells; lookup indicator. *Fig. 2.*
4. **Remedy:** held-out creativity across tasks and scales; per-seed deltas. *Fig. 3, Table 1.*
5. **What is necessary:** dose–response, TL weight, persistence, channel, regularizer baselines. *Fig. 4.*
6. **Mechanism:** interventions (prevention vs reversal), seed dependence (MI), probes. *Fig. 5.*
7. Related work: seed/prompt randomization, multi-token and teacherless training, minibatch-OT coupling,
   multiple-choice learning, label noise and regularization.
8. Limitations: algorithmic tasks, model scales, greedy vs sampling.

Appendix: optimal-coupling dynamics (freeze, gap, commitment); C01 compute; full per-seed tables; pre-registration
documents and incident register.
