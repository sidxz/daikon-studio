# Can the platform run a study like the SAC3 one?

**Last updated:** 2026-08-06. Companion to `roadmap.md` (what to build next) and
`engine-research.md` (external evidence). This document is the third axis: what a
*real, already-completed* study needed, and what the platform would have to grow to
host one.

The study: `/Users/sidx/workspace/inhibition-pred`. A model trained on ~407k in-house
Mtb ERA `% inhibition` compounds, applied prospectively to the SAC3 library (9,975
compounds), then compared against 960 wet-lab confirmed hits that arrived afterwards.
Artefacts on disk are results-only — the training repo is **not on this machine** and
is named nowhere, so parts of its method are reconstruction, marked below.

---

## Verdict

**No — about half, and it is the less interesting half.**

Training one model on that data is *nearly* in reach. Everything that made it a
**study** — combining models, replicating over seeds, and above all scoring a library
and then grading those predictions against wet-lab results that arrive weeks later —
is absent. There is one hard blocker underneath all of it, and it is ten lines wide.

> **Update, 2026-08-06.** The two missing *engines* were built on request:
> `ecfp4-lightgbm` and `molformer-xl`. The engine row of the table below is now
> complete; **nothing else in this document changed.** That is the point worth keeping
> in view — the roster was never the reason this study could not run here, and adding
> to it did not move the blocker. Both were measured on the real SAC3 library:
> `ecfp4-lightgbm` fits all 9,975 compounds in 9 s (test R² 0.18); `molformer-xl` on a
> frozen 2,000-compound subset reached test R² −0.004 in 42 s on `mps:0`, which is
> consistent with the study's own finding that MoLFormer is the weakest member at the
> sharp end and is a smoke test rather than a benchmark.

### The hard blocker

`RunPrediction` writes the results Parquet as
`{structure, <readouts>, uncertainty, applicability}` —
`application/execution/predict_with_protocol.py:322-350`. **Every other column of the
uploaded CSV is dropped.** The study joined predictions to wet-lab outcomes on
`Molecule Name` (SACC-xxxxxxx), not on SMILES. Today that join is impossible inside
the platform: the identifier never comes out the other side.

Nor can you fall back on row order. Unparseable structures are filtered *before* the
frame is written (`:283-286`), so `row_id` is a position in the results file, not in
the file the scientist uploaded. Upload 9,975 rows, get 9,970 back, with no way to
learn which five went missing.

Until an identifier survives a prediction run, **no prospective validation can be
built on top**, in the platform or outside it.

---

## Ingredient by ingredient

| The study used | daikon-studio today | |
|---|---|:--:|
| LightGBM on molecules | `ecfp4-lightgbm` (built 2026-08-06), plus the two XGBoost engines | ✅ |
| Chemprop D-MPNN | `chemprop-dmpnn`, ±CheMeleon | ✅ |
| Uni-Mol2 multi-task | ruled out — licence, audited leakage (`engine-research.md` §4) | ✅ skip |
| MolFormer-XL | `molformer-xl` (built 2026-08-06), frozen or fine-tuned | ✅ |
| Ridge stack over the 4 members | nothing — no ensembling of any kind | ❌ skip, see below |
| Within-seed std (`inhib_std`) | one seed, `dataset.split.seed`, no replication | ❌ |
| Cross-architecture range/std | needs N models' predictions side by side | ❌ |
| Train on 407k rows | 1800 s global timeout, whole frame in memory | ❌ |
| Score an external 9,975-compound library | prediction runs, cached, paginated | ✅ |
| **Join those predictions to compound IDs** | **only `structure` survives** | ❌ |
| Top-K hits caught / precision / enrichment | nothing | ❌ |
| Bottom-K enrichment | nothing | ❌ |
| Per-IC50-bucket recall | nothing | ❌ |
| AUROC at an 80% inhibition threshold | numeric target ⇒ only `rmse`/`mae`/`r2` | ⚠️ |
| Bootstrap CI on "which model won" | deferred by decision (`roadmap.md`) | ❌ |
| Buckets: top-1000 / bottom-50 / top-50 uncertain | sort + range filter + Collection export | ✅ |

Two entries need their ⚠️ explained.

**AUROC at a threshold** is a workaround, not a gap: pre-binarize `% inhib ≥ 80` into
its own column and declare a `BINARY` target. The cost is that you get either the
regressor or the threshold metrics from a run, never both, and the class balance is
brutal — **133 positives in 407,837 rows, 0.033%**. Worth stating plainly in any
writeup, because "top-100 enrichment ~89×" against a 0.033% base rate is ≈3 hits per
100 picks, which sounds very different from 89×.

**MolFormer-XL** stays skipped for *this* purpose on the study's own evidence, below.

### Where the platform is already ahead of the study

Not everything is a deficit. A run here ships a **mandatory baseline** fit on the
identical split, an **applicability-domain** number per predicted compound, an
**optimism gap** (a random re-split alongside every scaffold split), a **noise floor**
from duplicate spread, and per-scaffold error breakdowns. The study has none of these.

One concrete example: 23 SMILES strings (0.23%) are shared exactly between the 407k
training file and the SAC3 library, and nothing in the study addresses it. Those
compounds would come back with `applicability = 1.0` here and be visible on the
triage grid.

---

## What the study's own re-analysis says about ensembling

This is the reason the plan below builds no stacker.

Re-scoring the SAC3 library with each ensemble member separately, and with 11
rank-average combinations, reproduces the executive summary's headline row exactly
(8/17/30/55/109/184 hits at K = 10/50/100/250/500/1000) — so the comparison is on the
same footing. Then:

| Scorer | AUROC | top-10 | top-100 | top-1000 | potent <5 µM @1000 |
|---|---:|---:|---:|---:|---:|
| LightGBM alone | 0.520 | 8 | **37** | 173 | 28 |
| Chemprop D-MPNN | 0.520 | 3 | 35 | 186 | 30 |
| **Ridge stack (deployed)** | 0.535 | 8 | **30** | 184 | **35** |
| rank-avg LGBM+Chemprop | 0.522 | 7 | 31 | **199** | 31 |
| MolFormer-XL | 0.556 | 2 | 20 | 126 | 19 |
| Uni-Mol2 multi-task | 0.528 | 1 | 12 | 152 | 30 |

**The deployed stack is beaten at the sharp end by its own LightGBM member** — 37
against 30 at top-100, paired bootstrap over 2,000 compound resamples, median
difference +7, P(LightGBM > stack) = 0.95. No combination tried beat LightGBM at the
head. The stack's only win is potent-hit recall deep in the list (35 vs 28 @ top-1000).

Two members carry opposite pathologies, and both are informative:

- **Uni-Mol2** is the weakest member exactly where the model gets used — 1 hit in the
  top 10, 12 in the top 100 — and it is the trunk that absorbed a **220-GPU-hour LoRA
  fine-tune that ended at test R² 0.067 against a frozen baseline of 0.069.** The
  platform already refuses this model for licence and leakage reasons; this is
  independent corroboration from a different direction.
- **MolFormer-XL** has the best global AUROC (0.556) and near-useless top-100
  behaviour (20 hits). Including it *raises the number in the paper* while *lowering
  the number that matters*. Its position at #4 on the roadmap is fine on capability
  grounds; it is not a replication requirement.

So: **build the machinery that compares members on held-back outcomes; do not build
the stacker.** The comparison is what produced the finding. The stacker is what the
finding argues against.

---

## Reconstructed method (for whoever replicates it)

Recovered numerically from `predictions/sac3-lib_ranked.csv`; exact to float rounding.
The upstream training repo is missing, so this is what can be stated with evidence.

**The stack is a plain affine combination**, refit over all 9,975 rows at R² = 1.0,
max residual 2.2e-06:

```
inhib_pred = 0.788·lightgbm + 0.148·chemprop + 0.531·unimol_mt + 0.136·molformer − 1.648
```

Weights sum to 1.60 — not a convex blend, not a mean (a plain mean is off by up to
32.6 units). **What the ridge was fit on is unknown** — out-of-fold member predictions,
a held-out stacking set, or in-sample. If in-sample, the optimistic weights are a live
candidate explanation for why it loses out-of-domain on SAC3.

**Uncertainty columns:** `arch_range` = max − min over the four member predictions
(verified). `arch_std` = population std, `ddof=0`, over the same four (verified;
`ddof=1` is off by 5.6). `inhib_std` is **not** derivable from those four columns
(r = 0.62 with `arch_std`) — it is the within-seed spread, computed upstream.

**Buckets are fixed rank counts, not value thresholds** (all three verified as exact
set matches): `most_likely_positive` = top 1000 by prediction, `most_likely_negative`
= bottom 50, `high_uncertainty` = top 50 by `inhib_std`. They do not transfer to a
library of a different size without deciding whether to hold the count or the fraction
constant.

**The IC50 trap.** `Mtb_SAC3HitPlatesDR_Resazurin: IC50 (uM)` is object dtype: 664
numeric values, 2,472 rows of `'> 40.00'`, 1,080 rows of
`'(IC50 could not be calculated)'`. A naive numeric comparison raises `TypeError`; a
naive `read_csv` + filter silently drops 31% of hits. Coerce with
`pd.to_numeric(..., errors='coerce')`. Bucket edges are half-open and reproduce the
summary exactly: [0,5) = 120, [5,10) = 97, [10,20) = 118, [20,40) = 329.

**"Hit" means "carried into the IC50 follow-up plate"**, not "IC50 under a cutoff" —
every row of the hits file has a non-null IC50. 960 hits / 9,975 = 9.62% base rate.
The hits file holds 12 identical replicate rows per molecule, so the `groupby().min()`
dedup is a collapse, not an aggregation choice with teeth.

**Not recoverable, and load-bearing:** the split protocol behind "AUROC 0.82" (random
vs scaffold vs temporal, fold count, seeds), what the ridge was fit on, ridge α, every
member hyperparameter, and all preprocessing (the 407k file arrives pre-deduplicated
and pre-averaged — all three of `NATHAN ID` / `RU ID` / `Structure` are 407,837-unique).

---

## Plan

Four items, in dependency order. Items 1 and 2 are the study; 3 is what the study's
re-analysis says was worth more than the stack; 4 is the scale wall.

### 1. Carry an identifier through a prediction run

Everything else sits on this. Accept an optional `id_column` on the predict request
and copy it into the results Parquet beside `structure`; surface it on the results
page and in Collection exports. Ten lines and a request field.

Do the honest thing about dropped rows at the same time — a run that filters five
unparseable structures should say so, not silently return 9,970 rows.

### 2. Prospective validation: grade a run against outcomes that arrive later

The actual missing capability, and the one a wet-lab collaboration is built on.

Upload an outcomes CSV, point it at a finished prediction run, name the join column
and the outcome column (plus a threshold when the outcome is numeric). Get back: hits
caught, precision and enrichment over chance at each K; the same from the bottom of
the list; AUROC on the binary label; per-bucket recall when the outcome is graded; and
a paired bootstrap CI on every comparison, because at these counts the differences are
small and the platform already knows better than to print a naked delta
(`roadmap.md` — the unsupported "Beats the baseline" badge).

Not a Run. It is seconds of numpy over ~10⁴ rows — compute it on request. No queue, no
worker, no lane.

This also closes the ⚠️ on "AUROC at a threshold": binarizing a numeric outcome at a
caller-chosen cutoff is the same code path.

### 3. Prediction fan-out — the dual of sweeps

Sweeps already fan out N *training* runs over one dataset and rank them
(`application/execution/sweeps.py`). The missing symmetric move is fanning out N
*prediction* runs over one upload, and letting item 2 accept a set of runs so it emits
one row per protocol — the table in the section above, generated rather than
hand-rolled in a scratch script.

Three of the study's columns fall out of this one feature:

- **per-member comparison** — the finding that mattered;
- **cross-architecture range and std** — N columns side by side, which is all
  `arch_range` / `arch_std` ever were;
- **within-seed std** — the same config run N times, needing only that the seed become
  a training condition instead of being inherited from `dataset.split.seed`.

### 4. Per-lane job timeouts

Already carried as a trap in `roadmap.md`. One server-wide
`STUDIO_WORKER_JOB_TIMEOUT = 1800 s` covers every lane, which is why no sweep has yet
included a chemprop config — and why a chemprop fit on 407k molecules cannot start
here at all. The fix is per-lane timeouts, not avoiding the engine.

Watch the in-memory assumptions at that scale too: the whole training frame is read
per run (`train_protocol.py:451-453`), and the results Parquet is read in full on
every page request (`predict_with_protocol.py:537`, which already names
`scan_parquet` as its upgrade path). The 33 MB training file clears the 100 MB upload
cap, so the cap is not the wall — the deadline is.

### Deliberately not building

- **A stacker / ensemble combiner.** The study's own re-analysis says its deployed
  ridge stack lost to a single member at the sharp end (37 vs 30 @ top-100, P = 0.95).
  Build item 3, and revisit only when a fan-out shows a combination actually winning
  on held-back outcomes. That is a much better trigger than "ensembles are standard".
- **LightGBM.** A second gradient-boosting library for the same job as
  `descriptors-xgboost`. If a fan-out ever shows the gap is the library rather than
  the featurization, that is the moment.
- **Uni-Mol2, MolFormer-XL, multi-task.** Reasons above and in `engine-research.md`.
  None is required to replicate the useful part of this study.
