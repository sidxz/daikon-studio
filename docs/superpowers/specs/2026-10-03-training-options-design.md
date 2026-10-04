# Training options: tuned decision cutoffs, positive-class weighting, RDKit descriptor input

**Date:** 2026-10-03
**Status:** awaiting review
**Branch:** `training-options`, cut from `multi-task-labels` at `72b84db`. That branch is unmerged; this feature builds on its multi-target engine contract.

## Goal

Three options from the CAGE-Fusion paper's training recipe, built generically into the
app:

1. **Tuned decision cutoffs.** Each binary label gets a cutoff chosen to maximize MCC on the validation partition.
2. **Positive-class weighting.** Rare positives are up-weighted in the loss.
3. **RDKit descriptors as an extra model input.**

The paper is one motivating use, not the scope. Each option goes to every engine where the
method genuinely supports it, and to none where it would be faked. Chemprop is covered with
and without CheMeleon weights, on single- and multi-target datasets. Every option defaults
to off, so existing protocols and runs behave exactly as before.

## What exists today

- **The decision cutoff is a fixed 0.5, applied in four places:**
  - engine scoring (`classification_metrics` labels and sklearn `model.predict`);
  - the class column of prediction results (`predict_with_protocol.py`, with its own `ponytail:` note);
  - the Scorecard's bootstrap interval (`build_scorecard.primary_metric_ci`);
  - the model-versus-baseline comparison.
- **There is no class weighting anywhere.** "Class weights" was listed as out of scope in the multi-task spec.
- **RDKit descriptors** (`infrastructure/chem/featurize.rdkit_descriptors`, 217 descriptors with the installed RDKit) feed only the descriptors-XGBoost engine.
  - The values are raw.
  - Unmeasurable values are NaN.
  - Values outside float32 range are swept to NaN.
- **Engine settings** are `ConditionSpec`s declared per engine manifest. The frontend renders them generically. The baseline has its own conditions.
- **Engines compute their own metrics** with shared code in `infrastructure/engines/_scoring.py`. The application layer may not import infrastructure, so any metric logic must live where the metrics are computed.

## Decisions (user, 2026-10-03)

1. **The cutoff is a training option, not an engine setting.**
   - When it is on, the trained model, its baseline and the random-split comparison each get their own per-label cutoff.
   - Each cutoff is chosen on that fit's own validation predictions.
   - The model-versus-baseline comparison stays fair: tuned against tuned.
2. **Weighting and descriptors are generic.** They apply to every engine whose method supports them (table below), not only chemprop.
3. **Weighting modes are Off / Balanced / Square-root balanced.**
   - Balanced is the standard rule (negatives ÷ positives per label) and the paper's.
   - Square-root balanced is the common gentler variant for very rare labels.
   - Which is better depends on the data, so it is a sweepable setting rather than a fixed choice.
4. **Feature scaling applies in exactly one place: descriptors entering chemprop.**
   - Use a robust recipe: fill missing values with training means, apply a signed log, then standardize on training rows.
   - Trees, fingerprints, the GP and targets need no new scaling.

## Coverage

| Engine | Positive-class weighting | RDKit descriptors as extra input |
|---|---|---|
| Chemprop D-MPNN, with or without CheMeleon | ✓ custom loss, one weight per label | ✓ chemprop's extra-descriptor input, concatenated after the graph encoder |
| MoLFormer-XL | ✓ `BCEWithLogitsLoss(pos_weight=…)` | ✗ no input slot without an architecture change |
| ECFP4 + Random Forest | ✓ `class_weight={0: 1, 1: w}` | ✓ ECFP4 bits followed by the 217 descriptors |
| ECFP4 + XGBoost | ✓ `scale_pos_weight=w` | ✓ as above |
| ECFP4 + LightGBM | ✓ `scale_pos_weight=w` | ✓ as above |
| RDKit descriptors + XGBoost | ✓ `scale_pos_weight=w` | — already descriptors only |
| Tanimoto GP | ✗ sklearn's `GaussianProcessClassifier` takes no class weights | ✗ its kernel is defined on fingerprints |

The cutoff option applies to every engine, because every classifier in the roster
returns a probability.

## Design

### A. Tuned decision cutoffs (training option)

**Where it is set.**
- `TrainProtocolCommand.tune_cutoffs: bool = False` and `SubmitSweepCommand.tune_cutoffs: bool = False`.
- It is written to `runs.params` and included in the training `cache_key`.
- `TrainContext.tune_cutoffs: bool = False` carries it to engines.
- `RunTraining` passes the same value to all three fits: model, baseline and random split.

**The search.** `mcc_cutoff(y_true, probabilities) -> float | None` lives in `infrastructure/engines/_scoring.py`.
- The search is exact: every distinct validation probability is a candidate cutoff, and the rule is "predict positive when p ≥ c".
- It returns the candidate with the highest MCC.
- Ties are broken toward the higher cutoff, which predicts fewer positives and is the conservative choice.
- Implementation: sort once, then compute cumulative TP/FP counts, O(n log n).

**The guard.** It returns `None`, meaning keep 0.5, when the validation partition holds fewer than 10 positives or fewer than 10 negatives for the label.
- A cutoff picked from five positives fits noise.
- The reason is recorded and shown to the scientist.
- 10 is a module constant with a `ponytail:` note saying it could become configurable.

**Where it runs.** Inside engine scoring, through shared helpers in `_scoring.py`, so the cutoff and the MCC come from the same metric code.
- When `tune_cutoffs` is on, an engine scores its validation partition first.
- It picks one cutoff per binary target.
- It computes the threshold-dependent metrics (`mcc`, `balanced_accuracy`) on test and validation at that cutoff.
- `auroc` and `auprc` do not depend on the cutoff and are unchanged.
- Validation metrics at a tuned cutoff are optimistic, because the cutoff was chosen on them. The test metrics remain the verdict, and the Scorecard already labels validation as the tuning set.

**Contract.**
- `TrainResult.cutoffs: dict[str, float] | None`, keyed by target column. It holds only binary targets whose cutoff was tuned.
- `FanOut` merges per-target cutoffs.
- Joint engines return one cutoff per target.

**What is recorded.**
- `Readout` (domain.catalog) gains `threshold: float | None = None`, set on CLASS readouts. `None` means 0.5, which covers every protocol trained before this feature.
  - It is persisted in the `protocols.readouts` JSONB; old rows lack the key and read as `None`.
  - It also goes on `ReadoutWire` (defaulted, for old runners and old rows) and `ReadoutResponse`.
- `TargetInputs` (Scorecard inputs) gains:
  - `cutoff: float | None`, the model's;
  - `baseline_cutoff: float | None`;
  - `cutoff_note: str | None`, saying why a requested tune did not happen.
- These default to `None`, so legacy blobs read unchanged.
- `Scorecard` gains the same three fields.

**Where it is used.**
- **Prediction results:** the class column is `p ≥ (readout.threshold or 0.5)`. Exports and the triage grid follow automatically.
- **Scorecard:** the bootstrap interval for MCC is computed at the model's cutoff, so the point estimate and its interval describe the same classifier.

**UI.**
- **Train form and sweep form:** a "Tune decision cutoffs (maximize MCC on validation)" toggle, shown only when the dataset has a binary target, with one line saying it applies to the model and the baseline alike.
- **Scorecard verdict:** shows the cutoff next to MCC, e.g. "MCC 0.62 at cutoff 0.031, tuned on validation", and the baseline's own cutoff.
- **Protocol page:** the readouts list shows "class at cutoff 0.031" for tuned readouts.

### B. Positive-class weighting (engine setting)

**The setting.** One shared `ConditionSpec`, declared by each supporting engine:
- key `positive_weighting`, ENUM `none` / `balanced` / `sqrt_balanced`, default `none`;
- label "Positive-class weighting";
- help text explaining that it up-weights rare active compounds, that it inflates predicted probabilities, and that tuned cutoffs are recommended alongside it.

**Weights.**
- They are computed per binary label from the **training rows only**. Never validation or test, which would leak.
  - `balanced`: w = n_neg ÷ n_pos.
  - `sqrt_balanced`: w = √(n_neg ÷ n_pos).
- Dataset creation already guarantees n_pos ≥ 1 in training: a single-class training partition is refused.
- One helper, `positive_weight(y, mode) -> float`, lives in `_scoring.py`.

**Applicability.**
- `ConditionSpec` gains `tasks: tuple[TaskType, ...] = ()`, meaning the setting is only meaningful for these tasks; empty means all.
  - Weighting declares `(BINARY_CLASSIFICATION,)`.
  - The frontend hides a setting whose `tasks` does not intersect the dataset's tasks.
- Engines ignore weighting for regression fits. In a mixed dataset fanned out per target, the numeric targets are untouched and the binary ones are weighted. The help text says it applies to active/inactive targets only.

**Calibration.** Weighted models over-predict, so probabilities are no longer calibrated. The Scorecard's calibration curve will show this, which is the honest reading, not a defect.

### C. RDKit descriptors as extra input (engine setting)

**The setting.** One shared `ConditionSpec`:
- key `rdkit_descriptors`, BOOL, default `False`, label "Add RDKit descriptors";
- help text: "Adds 217 RDKit 2D descriptors (molecular weight, logP, TPSA, ring counts, …) to the model input."

**Descriptor set.**
- `featurize.DESCRIPTOR_NAMES`, the same list descriptors-XGBoost uses.
- The list is stored with every artifact that uses it.
- Prediction refuses with an actionable message if an RDKit upgrade has changed the list. This extends the existing `_require_matching_features` guard.

**Trees (RF, XGBoost, LightGBM).**
- The feature matrix is ECFP4 bits followed by the raw descriptors. No scaling: trees are invariant to order-preserving transforms.
- NaN is kept; all three handle missing values natively (sklearn ≥ 1.4 RF, verified on the installed 1.9).
- The artifact bundle records featurizer `"ecfp4+rdkit_descriptors"`. The predict path reads it from the bundle as it does today.

**Chemprop.** Preprocessing happens before the descriptors reach chemprop, because a neural net needs comparable input scales and RDKit descriptors are heavy-tailed:
1. Signed log: x → sign(x)·log(1+|x|). It preserves order, has no parameters, and tames outliers (an `Ipc` of 10³⁹ becomes about 90).
2. Fill missing values with the **training-row** mean of each (logged) descriptor. A descriptor missing in every training row is filled with 0.
3. Standardize with chemprop's native `normalize_inputs("X_d")` scaler, fitted on training rows and stored in the model as its `X_d_transform`.

The predictor's input width becomes graph-embedding width + 217. With CheMeleon that is 2048 + 217.

The training-mean fill values and the descriptor names are stored as extra keys in the checkpoint dictionary; Lightning ignores unknown keys on load.
- Predict detects descriptors from those keys.
- It recomputes the descriptors for the new compounds with the same three steps.
- Checkpoints without the keys predict exactly as today.

**Cost.** RDKit computes about 1–2 ms per molecule: roughly 15 s for 10k compounds and 5–10 min for 324k. Descriptors are computed once per fit, for all partitions together.

### Scaling, stated once

Scaling is needed only for descriptors entering chemprop (section C). Nothing changes for:
- tree inputs;
- ECFP4 bits;
- the Tanimoto GP;
- numeric targets, which are already standardized by chemprop, MoLFormer and the GP (`normalize_y`).

### Storage, cache keys and compatibility

- **No migration.** The new fields are an optional readout key, run params, and defaulted blob fields.
- **Cache key:** `tune_cutoffs` joins the training cache key. The new conditions are covered automatically, since conditions are already part of it.
- **Wire:** `ReadoutWire.threshold` is defaulted. As with the multi-task branch, deploy the API and runners together.
- **Legacy artifacts:**
  - pickled tree bundles without the new featurizer key default to ECFP4;
  - chemprop checkpoints without descriptor keys run unchanged;
  - MoLFormer bundles are unaffected, since weighting changes only training.

## Out of scope

- Learning-rate, optimizer or scheduler settings (AdamW, cosine with warmup).
- Hand-set per-label weights.
- Probability recalibration after weighting (Platt or isotonic).
- Balanced batch sampling (chemprop's `class_balance`).
- Descriptors for MoLFormer.
- Weighting for the GP.
- Sparse labels.
- Macro-averaged metrics.

## Testing

**Unit:**
- `mcc_cutoff` finds the exact optimum on a hand-built example, breaks ties upward, and returns `None` under the 10/10 guard.
- `positive_weight` returns n_neg/n_pos and its square root, from training rows only.
- The chemprop descriptor preprocessing round-trips: a training-fitted transform applied at predict reproduces training inputs.
- `ConditionSpec.tasks` hides weighting for numeric-only datasets.

**Engines:**
- Each supporting engine:
  - accepts `positive_weighting`; on an imbalanced toy set, Balanced raises the predicted-positive rate at 0.5 compared with none;
  - accepts `rdkit_descriptors` and changes its input width;
  - round-trips its artifact through predict.
- With `tune_cutoffs` on, `TrainResult.cutoffs` is set for binary targets and MCC is reported at that cutoff.
- Chemprop with CheMeleon, descriptors and weighting together trains on two targets and predicts.
- Legacy artifacts still predict.

**Integration:** a run with `tune_cutoffs` on records thresholds on CLASS readouts and cutoffs in Scorecard inputs. The baseline is tuned too. Prediction results' class column uses the threshold.

**Frontend:**
- the cutoff toggle on both forms;
- weighting hidden for numeric-only datasets;
- the Scorecard and readouts show cutoffs.

**Real data:** on the 10k nuisance sample, run one sweep with tuned cutoffs on four configurations:
- chemprop;
- chemprop with weighting and descriptors;
- CheMeleon;
- CheMeleon with weighting and descriptors.

Report per-label ROC-AUC, PR-AUC and MCC beside the paper's graph-only ablation row. This measures whether the options help on this data rather than assuming they do.

Gates (unchanged): ruff, ruff format, mypy strict, lint-imports (3 contracts), and `OMP_NUM_THREADS=1` pytest; frontend lint, tsc and vitest.
