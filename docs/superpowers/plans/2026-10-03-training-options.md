# Training Options Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three generic training options:
- per-label decision cutoffs tuned for MCC on validation (a training option, all engines);
- positive-class weighting (an engine setting, six engines);
- RDKit descriptors as an extra model input (an engine setting: chemprop and the three ECFP4 tree engines).

All default to off.

**Architecture:**
- Cutoffs are tuned **inside engine scoring**, so the cutoff and the MCC come from the same metric code. `TrainContext.tune_cutoffs` carries the request, and `TrainResult.cutoffs` returns per-target cutoffs.
- The cutoff is recorded on each CLASS `Readout` (`threshold`), so prediction results, exports and the triage grid follow automatically.
- Weighting and descriptors are shared `ConditionSpec`s, declared by each engine that supports them.
- Feature scaling applies only to descriptors entering chemprop: signed log, then the training-mean fill, then chemprop's own training-fitted scaler.

**Tech Stack:** Python 3.13, scikit-learn 1.9, XGBoost 3.3, LightGBM 4.7, chemprop 2.3 + Lightning, transformers (MoLFormer), polars, FastAPI, SQLAlchemy; Next.js, React Query, orval, Vitest.

**Spec:** `docs/superpowers/specs/2026-10-03-training-options-design.md`. Read it in full; this plan argues from it.

## Global Constraints

- **Branch:** `training-options`, cut from `multi-task-labels`. Do not switch branches.
- **Defaults are off:**
  - `tune_cutoffs=False`;
  - `positive_weighting="none"`;
  - `rdkit_descriptors=False`.

  With defaults, every engine must produce **bit-identical metrics and artifacts-behaviour** to today. With `cutoff=None`, keep the existing label path (`model.predict` for sklearn, `>= 0.5` for chemprop/MoLFormer).
- **Layers** (`lint-imports`, 3 contracts):
  - interface → infrastructure → application → domain;
  - the domain imports no numpy or polars;
  - `domain.catalog`, `domain.data` and `domain.execution` don't import each other.
  - Metric code lives in `infrastructure/engines/_scoring.py`. Application code may not import it, so shared *constants* go in the application layer.
- **Chemprop import discipline:** chemprop, torch and lightning are imported only **inside functions** in `chemprop_dmpnn.py`; `default_registry()` instantiates the engine on API workers that have no chemprop. The new `_chemprop_loss.py` imports chemprop at module top, so it must be imported only from inside chemprop functions.
- **Leakage:** every fitted statistic is computed from **training rows only**: positive weights, descriptor fill means, the descriptor scaler. The cutoff is fitted on **validation rows only**.
- **Copy:** American spelling, academic and plain, no em-dash clause chains. Use the strings given here.
- **Tests per task:** run only the test files you touch plus the gates:
  ```
  uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src && uv run lint-imports
  ```
  Use `OMP_NUM_THREADS=1`. **No full suites.**
- **Known pre-existing ordering problems:** run torch-loading engine tests (chemprop, molformer) in their own pytest invocation, separate from `test_ecfp4_lightgbm.py` (GitHub issue #1). Don't run `tests/integration/test_sweeps.py` and `tests/api/test_runner_protocol.py` in one process.
- **Commits:**
  - conventional subject;
  - end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`;
  - **never** a `Claude-Session:` trailer;
  - stage explicit paths;
  - no `git stash`.
- **Runners don't hot-reload.** Before any live check, run `make dev-worker` and `make dev-worker-gpu`.

## Review Focus

1. **Defaults off must change nothing.** Every engine's existing tests must pass unmodified, apart from those asserting resolved-condition dicts, which now include the two new keys. Pinned in T2, T3 and T4: the existing test files pass, and one test per engine family asserts the default path still uses `model.predict` / `>= 0.5`.
2. **Double-scaled validation descriptors in chemprop.** Chemprop's own CLI normalizes the validation dataset *and* installs `X_d_transform`, which applies again in eval mode, so validation gets scaled twice. We scale only the training set. Pinned in T3: a test that the descriptor transform round-trips (training-fitted fill and scale reproduce the training inputs), plus a code comment at the call site.
3. **Old artifacts after the change.**
   - pickled tree bundles without a `featurizer` key;
   - chemprop checkpoints without `daikon_descriptors`;
   - checkpoints whose criterion is the stock `BCELoss`.

   All must still predict. Pinned in T2/T3 via the existing legacy tests, plus a weighted-checkpoint save → load → predict test.
4. **A tuned cutoff from too few validation positives.** For example, `reactive`, with 27 actives in 1,000 validation rows. The cutoff must stay 0.5 below 10 per class, and the Scorecard must say why. Pinned in T1 (guard) and T5 (note).
5. **Tuned model compared against an untuned baseline.** The same `tune_cutoffs` must reach all three fits. Pinned in T5: the integration test asserts `baseline_cutoff` is set when `cutoff` is.

---

### Task 1: Shared building blocks

**Files:**
- Modify: `backend/src/daikonstudio/application/engines/manifest.py` (`ConditionSpec.tasks`)
- Modify: `backend/src/daikonstudio/application/engines/context.py`:
  - `MIN_CUTOFF_CLASS_COUNT`;
  - `TrainContext.tune_cutoffs`;
  - `TrainResult.cutoffs`.
- Modify: `backend/src/daikonstudio/application/engines/fan_out.py` (merge cutoffs)
- Create: `backend/src/daikonstudio/infrastructure/engines/_options.py`
- Modify: `backend/src/daikonstudio/infrastructure/engines/_scoring.py`:
  - `mcc_cutoff`;
  - the combined featurizer;
  - the feature-names entry.
- Modify: `backend/src/daikonstudio/interface/routes/engines.py` (`ConditionResponse.tasks`)
- Test:
  - `backend/tests/unit/engines/test_options.py` (create);
  - `backend/tests/unit/engines/test_fan_out.py`;
  - `backend/tests/api/test_engines.py`.

**Interfaces produced:**
- `ConditionSpec.tasks: tuple[TaskType, ...] = ()`
- `MIN_CUTOFF_CLASS_COUNT: int = 10` (in `application/engines/context.py`)
- `TrainContext.tune_cutoffs: bool = False`
- `TrainResult.cutoffs: dict[str, float] | None = None`
- `POSITIVE_WEIGHTING`, `RDKIT_DESCRIPTORS` (`ConditionSpec`s)
- `positive_weight(y_train: np.ndarray, mode: str) -> float | None`
- `mcc_cutoff(y_true: np.ndarray, probabilities: np.ndarray) -> float | None`
- `tree_featurizer(conditions: dict[str, Any]) -> tuple[str, Featurizer]`
- `bundle_features(featurizer_key: str) -> dict[str, Any]`
- `ConditionResponse.tasks: list[str]`

- [ ] **Step 1: Failing tests**

Create `backend/tests/unit/engines/test_options.py`:

```python
import math

import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef

from daikonstudio.application.engines.context import MIN_CUTOFF_CLASS_COUNT
from daikonstudio.infrastructure.chem.featurize import DESCRIPTOR_NAMES
from daikonstudio.infrastructure.engines._options import positive_weight
from daikonstudio.infrastructure.engines._scoring import (
    _FEATURIZERS,
    mcc_cutoff,
    tree_featurizer,
)


def _reference_cutoff(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Brute force: every distinct probability, predict positive when p >= c."""
    best = (-2.0, 0.0)
    for c in sorted(set(p.tolist()), reverse=True):
        mcc = matthews_corrcoef(y, (p >= c).astype(int))
        if mcc > best[0]:
            best = (mcc, c)
    return best


def test_mcc_cutoff_finds_the_exact_optimum():
    rng = np.random.default_rng(7)
    y = (rng.random(300) < 0.15).astype(int)
    p = np.clip(0.15 + 0.4 * y + rng.normal(0, 0.2, 300), 0, 1)
    best_mcc, best_cut = _reference_cutoff(y, p)
    cut = mcc_cutoff(y, p)
    assert cut == pytest.approx(best_cut)
    assert matthews_corrcoef(y, (p >= cut).astype(int)) == pytest.approx(best_mcc)


def test_mcc_cutoff_breaks_ties_toward_the_higher_cutoff():
    y = np.array([1] * 10 + [0] * 10)
    p = np.array([0.9] * 10 + [0.2] * 5 + [0.1] * 5)
    # Any cutoff in (0.2, 0.9] separates perfectly; the candidates are 0.9, 0.2, 0.1.
    assert mcc_cutoff(y, p) == pytest.approx(0.9)


def test_mcc_cutoff_refuses_too_few_of_either_class():
    y = np.array([1] * (MIN_CUTOFF_CLASS_COUNT - 1) + [0] * 50)
    p = np.linspace(0, 1, len(y))
    assert mcc_cutoff(y, p) is None


def test_positive_weight_uses_the_training_ratio():
    y = np.array([1] * 4 + [0] * 36)
    assert positive_weight(y, "none") is None
    assert positive_weight(y, "balanced") == pytest.approx(9.0)
    assert positive_weight(y, "sqrt_balanced") == pytest.approx(3.0)
    assert positive_weight(np.zeros(10), "balanced") is None


def test_tree_featurizer_appends_the_descriptors_as_float32():
    key, featurize = tree_featurizer({"rdkit_descriptors": True})
    x = featurize(["CCO", "c1ccccc1"])
    assert key == "ecfp4+rdkit_descriptors"
    assert x.shape == (2, 2048 + len(DESCRIPTOR_NAMES))
    assert x.dtype == np.float32
    assert _FEATURIZERS[key] is featurize
    assert tree_featurizer({"rdkit_descriptors": False})[0] == "ecfp4"
    assert tree_featurizer({})[0] == "ecfp4"
```

Append to `backend/tests/unit/engines/test_fan_out.py`. Extend `_Recorder.train` so its `TrainResult` carries `cutoffs={column: 0.3}` when `ctx.tune_cutoffs` is set, then add:

```python
def test_cutoffs_merge_per_target_and_tune_cutoffs_reaches_every_sub_fit():
    inner = _Recorder()
    result = FanOut(inner).train(
        replace(
            _ctx({"a": TaskType.BINARY_CLASSIFICATION, "b": TaskType.BINARY_CLASSIFICATION}),
            tune_cutoffs=True,
        )
    )
    assert result.cutoffs == {"a": 0.3, "b": 0.3}
    assert all(ctx.tune_cutoffs for ctx in inner.contexts)
```

Add `from dataclasses import replace` if missing. In `backend/tests/api/test_engines.py`, assert that the chemprop engine's `positive_weighting` condition lists `"binary_classification"` in `tasks`. That assertion goes green in Task 3, which declares the setting on chemprop; for now assert that `tasks` exists (`== []`) on an existing condition.

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_options.py tests/unit/engines/test_fan_out.py -q
```

Expected: FAIL with import errors.

- [ ] **Step 3: Implement**

**`manifest.py`**, in `ConditionSpec` after `help`:

```python
    # The tasks this setting means anything for; empty means every task. A form hides a
    # setting the dataset has no such task for, and an engine ignores it in a fit of any
    # other task -- one target of a mixed dataset, fanned out.
    tasks: tuple[TaskType, ...] = ()
```

**`context.py`:**

```python
#: Fewer validation positives or negatives than this for a label and a cutoff tuned on
#: them fits noise: five actives pick whichever cutoff happens to separate those five.
#: Defined here, not beside the search in `_scoring.py`, because the training run quotes
#: it when it explains an untuned cutoff, and application code may not import engines.
#: ponytail: fixed at 10 per class; make it a training setting if a rarer label needs it.
MIN_CUTOFF_CLASS_COUNT = 10
```

`TrainContext` gains, before `report`:

```python
    # Choose each binary target's decision cutoff to maximize MCC on the validation
    # partition, and report the threshold-dependent metrics at it. The training run sets
    # the same value for the model, its baseline and the random-split comparison, so a
    # tuned model is never measured against an untuned baseline.
    tune_cutoffs: bool = False
```

`TrainResult` gains:

```python
    # Each binary target's tuned cutoff, keyed by column: only targets whose cutoff was
    # tuned (`tune_cutoffs` on, and validation held enough of both classes).
    cutoffs: dict[str, float] | None = None
```

**`fan_out.py`**, in `train`:

```python
        cutoffs = {
            key: value
            for result in results
            if result.cutoffs is not None
            for key, value in result.cutoffs.items()
        }
```

Pass `cutoffs=cutoffs or None` to the returned `TrainResult`. The `replace(ctx, …)` per sub-fit already carries `tune_cutoffs`.

**Create `infrastructure/engines/_options.py`:**

```python
"""Training settings more than one engine declares, and the arithmetic behind them.

Declared once so every engine that supports a setting offers it with the same key, the
same options and the same words -- a sweep comparing engines compares like with like.
"""

from __future__ import annotations

import math

import numpy as np

from daikonstudio.application.engines.manifest import ConditionSpec, ConditionType, TaskType

POSITIVE_WEIGHTING = ConditionSpec(
    key="positive_weighting",
    label="Positive-class weighting",
    type=ConditionType.ENUM,
    default="none",
    options=("none", "balanced", "sqrt_balanced"),
    tasks=(TaskType.BINARY_CLASSIFICATION,),
    help=(
        "Up-weights active compounds in the loss so a rare label is not drowned out. "
        "Balanced weights each label's actives by its inactive-to-active ratio in the "
        "training set; square-root balanced uses the square root of that ratio, a gentler "
        "choice for very rare labels. Weighting inflates predicted probabilities, so tune "
        "decision cutoffs alongside it. Applies to active/inactive targets only."
    ),
)

RDKIT_DESCRIPTORS = ConditionSpec(
    key="rdkit_descriptors",
    label="Add RDKit descriptors",
    type=ConditionType.BOOL,
    default=False,
    help=(
        "Adds RDKit's 2D descriptor set (about 200 descriptors, such as molecular weight, "
        "logP, TPSA and ring counts) to the model's input."
    ),
)


def positive_weight(y_train: np.ndarray, mode: str) -> float | None:
    """The loss weight on one binary label's positives, from its training labels only.

    `None` when weighting is off, or when the training rows hold no positives -- possible
    only in the random-split comparison's reshuffle, since dataset creation refuses a
    single-class training partition.
    """
    if mode == "none":
        return None
    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    if positives == 0:
        return None
    ratio = negatives / positives
    return ratio if mode == "balanced" else math.sqrt(ratio)
```

**`_scoring.py`:**
- Add `from daikonstudio.application.engines.context import MIN_CUTOFF_CLASS_COUNT`.
- Add `rdkit_descriptors` and `DESCRIPTOR_NAMES` to the featurize import (they're already imported).
- Add:

```python
def mcc_cutoff(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    """The cutoff maximizing MCC when "positive" means p >= cutoff, or None.

    Exact, not a grid: every distinct probability is a candidate, so a label whose
    probabilities all sit below 0.05 still gets a meaningful cutoff. Ties go to the
    higher cutoff, which predicts fewer positives -- the conservative reading. `None`
    when either class has fewer than `MIN_CUTOFF_CLASS_COUNT` members.
    """
    y = np.asarray(y_true) == 1
    p = np.asarray(probabilities, dtype=np.float64)
    positives = float(y.sum())
    negatives = float(len(y)) - positives
    if positives < MIN_CUTOFF_CLASS_COUNT or negatives < MIN_CUTOFF_CLASS_COUNT:
        return None
    order = np.argsort(-p, kind="stable")
    p_sorted, y_sorted = p[order], y[order]
    tp = np.cumsum(y_sorted, dtype=np.float64)
    fp = np.cumsum(~y_sorted, dtype=np.float64)
    # One candidate per distinct value: cut after its last occurrence in sorted order.
    last = np.r_[p_sorted[1:] != p_sorted[:-1], True]
    tp, fp, cuts = tp[last], fp[last], p_sorted[last]
    fn, tn = positives - tp, negatives - fp
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.full_like(denominator, -np.inf)
    defined = denominator > 0
    mcc[defined] = (tp[defined] * tn[defined] - fp[defined] * fn[defined]) / denominator[defined]
    # argmax returns the first maximum; candidates run from the highest cutoff down.
    return float(cuts[int(np.argmax(mcc))])


def _ecfp4_with_descriptors(smiles_list: list[str]) -> np.ndarray:
    """ECFP4 bits followed by the raw RDKit descriptors, as float32.

    Raw because trees split on thresholds and are indifferent to scale; NaN kept because
    all three tree learners route missing values natively. float32 because that is what
    the learners train on internally, and the descriptor featurizer already keeps every
    value inside float32 range.
    """
    return np.hstack(
        [ecfp4(smiles_list).astype(np.float32), rdkit_descriptors(smiles_list).astype(np.float32)]
    )
```

Register it, then add the helpers:

```python
_FEATURIZERS["ecfp4+rdkit_descriptors"] = _ecfp4_with_descriptors
_FEATURE_NAMES["ecfp4+rdkit_descriptors"] = DESCRIPTOR_NAMES


def tree_featurizer(conditions: dict[str, Any]) -> tuple[str, Featurizer]:
    """The bundle key and featurizer an ECFP4 tree engine fits with, from its settings."""
    key = "ecfp4+rdkit_descriptors" if conditions.get("rdkit_descriptors") else "ecfp4"
    return key, _FEATURIZERS[key]


def bundle_features(featurizer_key: str) -> dict[str, Any]:
    """What a tree bundle records about its input, so predict featurizes identically and
    `_require_matching_features` can refuse a descriptor list that has since changed."""
    names = _FEATURE_NAMES.get(featurizer_key)
    return {"featurizer": featurizer_key} | ({"feature_names": names} if names else {})
```

**`routes/engines.py`:**
- `ConditionResponse` gains `tasks: list[str]`, with a comment: the tasks the setting applies to; empty means all.
- `from_spec` sets `tasks=[task.value for task in spec.tasks]`.

- [ ] **Step 4: Run tests and gates, then commit**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_options.py tests/unit/engines/test_fan_out.py tests/api/test_engines.py -q && uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src && uv run lint-imports
```

Expected: PASS.

```bash
git add backend/src/daikonstudio/application/engines backend/src/daikonstudio/infrastructure/engines/_options.py backend/src/daikonstudio/infrastructure/engines/_scoring.py backend/src/daikonstudio/interface/routes/engines.py backend/tests/unit/engines/test_options.py backend/tests/unit/engines/test_fan_out.py backend/tests/api/test_engines.py
git commit -m "feat(engines): shared cutoff search, positive weights and descriptor features

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Tree engines and the GP

**Files:**
- Modify: `backend/src/daikonstudio/infrastructure/engines/_scoring.py`. Replace `_score` and `_score_validation` with `_scored`; `_metrics_on` takes features, labels and a cutoff.
- Modify:
  - `ecfp4_randomforest.py`;
  - `ecfp4_xgboost.py`;
  - `ecfp4_lightgbm.py`;
  - `descriptors_xgboost.py`;
  - `tanimoto_gp.py`.

  All live in `backend/src/daikonstudio/infrastructure/engines/`.
- Test:
  - `backend/tests/unit/engines/test_ecfp4_engines.py`;
  - `test_ecfp4_lightgbm.py`;
  - `test_descriptors_xgboost.py`;
  - `test_tanimoto_gp.py`.

**Interfaces:**
- Consumes (Task 1): `mcc_cutoff`, `positive_weight`, `tree_featurizer`, `bundle_features`, `POSITIVE_WEIGHTING`, `RDKIT_DESCRIPTORS`, `TrainContext.tune_cutoffs`, `TrainResult.cutoffs`.
- Produces: `_scored(model, ctx, is_classification, featurizer=ecfp4) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]] | None, dict[str, float] | None]`.

**Engine coverage:**

| Engine | `positive_weighting` | `rdkit_descriptors` | Cutoff tuning |
|---|---|---|---|
| RF | ✓ | ✓ | ✓ |
| XGBoost | ✓ | ✓ | ✓ |
| LightGBM | ✓ | ✓ | ✓ |
| descriptors-XGBoost | ✓ | ✗ | ✓ |
| GP | ✗ | ✗ | ✓ |

- [ ] **Step 1: Failing tests**

Add a shared helper to `tests/unit/engines/test_ecfp4_engines.py`:

```python
def _imbalanced_frame() -> pl.DataFrame:
    """300 rows: alcohols inactive, amines active (about 20%), deterministic split."""
    alcohols = [f"{'C' * n}O" for n in range(1, 121)]
    amines = [f"{'C' * n}N" for n in range(1, 31)]
    smiles = (alcohols + amines) * 2
    labels = ([0] * len(alcohols) + [1] * len(amines)) * 2
    split = ["train", "validation", "test", "train", "train"] * (len(smiles) // 5)
    return pl.DataFrame({"smiles": smiles, "y": labels, "split": split})
```

Then tests, parametrized over `Ecfp4RandomForest()`, `Ecfp4XGBoost()`, `Ecfp4LightGBM()` (and `DescriptorsXGBoost()` for weighting only):

```python
def _binary_ctx(frame, **conditions):
    return TrainContext(
        frame=frame,
        targets={"y": TaskType.BINARY_CLASSIFICATION},
        structure_column="smiles",
        conditions=conditions,
        seed=1,
    )


@pytest.mark.parametrize("engine", [Ecfp4RandomForest(), Ecfp4XGBoost(), Ecfp4LightGBM()])
def test_balanced_weighting_raises_the_predicted_probability_of_actives(engine):
    frame = _imbalanced_frame()
    test = frame.filter(pl.col("split") == "test")

    def mean_probability(**conditions):
        artifact = engine.train(_binary_ctx(frame, **conditions)).artifact
        out = engine.predict(
            PredictContext(
                frame=test, structure_column="smiles", artifact=artifact, conditions={},
                target_columns=("y",),
            )
        )
        return out["value"].mean()

    assert mean_probability(positive_weighting="balanced") > mean_probability()


@pytest.mark.parametrize("engine", [Ecfp4RandomForest(), Ecfp4XGBoost(), Ecfp4LightGBM()])
def test_descriptors_widen_the_input_and_round_trip(engine):
    frame = _imbalanced_frame()
    result = engine.train(_binary_ctx(frame, rdkit_descriptors=True))
    bundle = pickle.loads(result.artifact)
    assert bundle["featurizer"] == "ecfp4+rdkit_descriptors"
    assert bundle["model"].n_features_in_ == 2048 + len(DESCRIPTOR_NAMES)
    out = engine.predict(
        PredictContext(
            frame=frame.head(5), structure_column="smiles", artifact=result.artifact,
            conditions={}, target_columns=("y",),
        )
    )
    assert out.height == 5


@pytest.mark.parametrize("engine", [Ecfp4RandomForest(), Ecfp4XGBoost(), Ecfp4LightGBM()])
def test_a_tuned_cutoff_is_returned_and_mcc_is_reported_at_it(engine):
    frame = _imbalanced_frame()
    result = engine.train(replace(_binary_ctx(frame), tune_cutoffs=True))
    assert result.cutoffs is not None and set(result.cutoffs) == {"y"}
    cut = result.cutoffs["y"]
    test = frame.filter(pl.col("split") == "test")
    probabilities = engine.predict(
        PredictContext(
            frame=test, structure_column="smiles", artifact=result.artifact, conditions={},
            target_columns=("y",),
        )
    )["value"].to_numpy()
    expected = matthews_corrcoef(test["y"].to_numpy(), (probabilities >= cut).astype(int))
    assert result.metrics["y"]["mcc"] == pytest.approx(expected)


@pytest.mark.parametrize("engine", [Ecfp4RandomForest(), Ecfp4XGBoost(), Ecfp4LightGBM()])
def test_without_tuning_there_are_no_cutoffs(engine):
    assert engine.train(_binary_ctx(_imbalanced_frame())).cutoffs is None
```

Add imports as needed: `pickle`, `replace`, `matthews_corrcoef`, `DESCRIPTOR_NAMES`, `Ecfp4LightGBM`, `DescriptorsXGBoost`. LightGBM lives in its own file; its tests can sit there to keep the existing per-file layout. Add one tuned-cutoff test for `TanimotoGP` in `test_tanimoto_gp.py` (it's slow, so use 120 rows), and one weighting test for `DescriptorsXGBoost` in its file.

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_ecfp4_engines.py -q
```

Expected: FAIL. The conditions are unknown (`does not accept these settings`) and there are no cutoffs.

- [ ] **Step 3: `_scored` in `_scoring.py`**

Replace `_metrics_on`, `_score` and `_score_validation` with:

```python
def _metrics_on(
    model: Any, x: np.ndarray, y: np.ndarray, is_classification: bool, cutoff: float | None
) -> dict[str, float]:
    """RMSE/MAE/R2 for regression; MCC/balanced accuracy/AUROC/AUPRC for classification.

    With no cutoff the hard labels are `model.predict` exactly as before; with one they
    are `p >= cutoff`, so MCC and balanced accuracy describe the tuned classifier while
    AUROC and AUPRC, which no cutoff touches, are unchanged.
    """
    if not is_classification:
        return regression_metrics(y, model.predict(x))
    # (keep the existing comment about short-circuiting before any sklearn call)
    if len(np.unique(y)) < 2 or len(model.classes_) < 2:
        return _undefined_classification_metrics()
    probabilities = _positive_class_probability(model, x)
    labels = model.predict(x) if cutoff is None else (probabilities >= cutoff).astype(int)
    return classification_metrics(y, labels, probabilities, train_has_both_classes=True)


def _scored(
    model: Any, ctx: TrainContext, is_classification: bool, featurizer: Featurizer = ecfp4
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]] | None, dict[str, float] | None]:
    """Test metrics, validation metrics and the tuned cutoff for the one target this fit
    trained on, keyed by its column -- the shapes `TrainResult` takes.

    Validation is scored first because, when `ctx.tune_cutoffs` is on, it is where the
    cutoff comes from; test is then scored at that cutoff. Validation metrics at a tuned
    cutoff are optimistic, since the cutoff was chosen on them -- the test numbers stay
    the verdict. Each partition is featurized once.
    """
    column = ctx.target_column
    test_rows = ctx.frame.filter(pl.col("split") == "test")
    validation_rows = ctx.frame.filter(pl.col("split") == "validation")
    x_validation = (
        featurizer(validation_rows[ctx.structure_column].to_list())
        if validation_rows.height
        else None
    )
    cutoff = None
    if (
        ctx.tune_cutoffs
        and is_classification
        and x_validation is not None
        and len(model.classes_) == 2
    ):
        cutoff = mcc_cutoff(
            validation_rows[column].to_numpy(), _positive_class_probability(model, x_validation)
        )
    x_test = featurizer(test_rows[ctx.structure_column].to_list())
    metrics = {
        column: _metrics_on(model, x_test, test_rows[column].to_numpy(), is_classification, cutoff)
    }
    validation = (
        None
        if x_validation is None
        else {
            column: _metrics_on(
                model, x_validation, validation_rows[column].to_numpy(), is_classification, cutoff
            )
        }
    )
    return metrics, validation, ({column: cutoff} if cutoff is not None else None)
```

- [ ] **Step 4: Engines**

**Every tree engine and the GP:** replace
`metrics=_score(...), validation_metrics=_score_validation(...)` with:

```python
        metrics, validation_metrics, cutoffs = _scored(model, ctx, is_classification, featurizer)
        return TrainResult(
            artifact=artifact,
            metrics=metrics,
            validation_metrics=validation_metrics,
            cutoffs=cutoffs,
        )
```

The GP and descriptors-XGBoost pass their own featurizer, as they did to `_score`.

**RF, XGBoost and LightGBM:**
- Append `POSITIVE_WEIGHTING, RDKIT_DESCRIPTORS` to the manifest's `conditions`.
- In `train`:

```python
        featurizer_key, featurizer = tree_featurizer(conditions)
        x_train = featurizer(train_rows[ctx.structure_column].to_list())
        y_train = train_rows[ctx.target_column].to_numpy()
        is_classification = ctx.task is TaskType.BINARY_CLASSIFICATION
        # Regression fits ignore the setting: in a mixed dataset fanned out per target,
        # only the active/inactive targets are weighted.
        weight = (
            positive_weight(y_train, str(conditions["positive_weighting"]))
            if is_classification
            else None
        )
```

- RF: `RandomForestClassifier(..., class_weight=None if weight is None else {0: 1.0, 1: weight})`.
- XGBoost and LightGBM: `if weight is not None: model_kwargs["scale_pos_weight"] = weight`, for the classifier only.
- Bundle: `pickle.dumps({"model": model, "is_classification": is_classification, **bundle_features(featurizer_key)})`.

**descriptors-XGBoost:**
- Append `POSITIVE_WEIGHTING`.
- Compute `weight` the same way and set `scale_pos_weight`.
- Its bundle already records its featurizer.

**GP:** no new conditions; only `_scored`.

- [ ] **Step 5: Update tests that pin resolved conditions or the removed helpers**

Run:
```
cd backend && grep -rn "_score(\|_score_validation\|_metrics_on(" tests src
```

Update each hit. Resolved-condition dicts in tests now include `"positive_weighting": "none"` and `"rdkit_descriptors": False` for the engines that declare them.

- [ ] **Step 6: Run tests and gates, then commit**

Run (two invocations, issue #1):
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_ecfp4_engines.py tests/unit/engines/test_descriptors_xgboost.py tests/unit/engines/test_tanimoto_gp.py tests/unit/engines/test_options.py -q && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_ecfp4_lightgbm.py -q && uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src && uv run lint-imports
```

Expected: PASS. Existing tests in those files pass unmodified, except resolved-condition dicts.

```bash
git add backend/src/daikonstudio/infrastructure/engines backend/tests/unit/engines
git commit -m "feat(engines): tree engines and the GP tune cutoffs, weight positives, add descriptors

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Chemprop, with and without CheMeleon

**Files:**
- Create: `backend/src/daikonstudio/infrastructure/engines/_chemprop_loss.py`
- Modify: `backend/src/daikonstudio/infrastructure/engines/chemprop_dmpnn.py`
- Test: `backend/tests/unit/engines/test_chemprop_dmpnn.py`

**Interfaces:**
- Consumes (Task 1): `POSITIVE_WEIGHTING`, `RDKIT_DESCRIPTORS`, `positive_weight`, `mcc_cutoff`, `TrainContext.tune_cutoffs`, `TrainResult.cutoffs`.
- Produces:
  - `PositiveWeightedBCELoss(pos_weight: list[float])`;
  - `_descriptor_inputs(raw: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]`, returning `(x_d, fill)`;
  - checkpoint key `"daikon_descriptors": {"names": [...], "fill": [...]}`.

- [ ] **Step 1: Failing tests**

Add to `tests/unit/engines/test_chemprop_dmpnn.py`, reusing that file's chemprop skip guard:

```python
def test_descriptor_inputs_fill_from_training_rows_only():
    from daikonstudio.infrastructure.engines.chemprop_dmpnn import _descriptor_inputs

    raw = np.array([[1.0, np.nan], [3.0, np.nan], [np.nan, 5.0], [1e39, np.nan]])
    train = np.array([True, True, False, False])
    x_d, fill = _descriptor_inputs(raw, train)
    logged = np.sign(raw) * np.log1p(np.abs(raw))
    assert fill[0] == pytest.approx(np.mean(logged[:2, 0]))  # training rows only
    assert fill[1] == 0.0  # missing in every training row
    assert x_d[2, 0] == pytest.approx(fill[0])  # a non-training gap gets the training fill
    assert x_d[3, 0] == pytest.approx(np.log1p(1e39))  # heavy tail tamed, about 90
    assert np.isfinite(x_d).all()


@pytest.mark.parametrize("pretrained", ["none", "CheMeleon"])
def test_weighting_descriptors_and_cutoffs_train_jointly_and_predict(pretrained):
    frame = _two_binary_targets_frame()  # build: >= 300 rows, >= 10 actives per label in validation
    engine = ChempropDMPNN()
    result = engine.train(
        TrainContext(
            frame=frame,
            targets={"a": TaskType.BINARY_CLASSIFICATION, "b": TaskType.BINARY_CLASSIFICATION},
            structure_column="smiles",
            conditions={
                "epochs": 2,
                "pretrained": pretrained,
                "positive_weighting": "balanced",
                "rdkit_descriptors": True,
            },
            seed=1,
            tune_cutoffs=True,
        )
    )
    assert result.cutoffs is not None and set(result.cutoffs) <= {"a", "b"}
    out = engine.predict(
        PredictContext(
            frame=frame.head(8), structure_column="smiles", artifact=result.artifact,
            conditions={}, target_columns=("a", "b"),
        )
    )
    assert out.height == 16
    assert out["value"].is_finite().all()


def test_a_changed_descriptor_list_is_refused_at_predict():
    # train with rdkit_descriptors=True (2 epochs, small frame), then rewrite the stored
    # names in the checkpoint dict (torch.load / torch.save) and expect ValidationError
    # mentioning "descriptor" from predict.
    ...
```

Write the last test out fully, as its comment describes.

`_two_binary_targets_frame()` builds alcohols and amines like Task 2's `_imbalanced_frame`:
- column `a` is 1 for amines;
- column `b` is 1 for odd chain lengths;
- split `["train", "validation", "test", "train", "train"]` repeating;
- size chosen so validation holds at least 10 of each class per label.

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_chemprop_dmpnn.py -q -rs
```

Expected: FAIL, not skipped. Chemprop and the CheMeleon weights are installed in the dev venv; a SKIP is not a pass.

- [ ] **Step 3: The weighted loss module**

Create `_chemprop_loss.py`:

```python
"""The positive-class-weighted loss chemprop trains with when `positive_weighting` is on.

A module of its own, importing chemprop at the top, because a chemprop checkpoint
pickles its criterion by reference: loading the model must be able to import this class
by its qualified name. Only `chemprop_dmpnn.py` imports this module, and only inside
functions, so the API tier and the default-lane runner -- which have no chemprop -- never
do.
"""

from __future__ import annotations

import torch
from chemprop.nn.metrics import BCELoss
from torch import Tensor
from torch.nn import functional as F


class PositiveWeightedBCELoss(BCELoss):  # type: ignore[misc]
    """chemprop's BCE with one positive-class weight per task, applied inside the
    logit-space loss exactly as `torch.nn.BCEWithLogitsLoss(pos_weight=...)` does."""

    def __init__(self, pos_weight: list[float], **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args: object) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            preds, targets, pos_weight=self.pos_weight, reduction="none"
        )
```

Check how `BCELoss.__init__` takes `task_weights` in chemprop 2.3 (`inspect.signature`) and pass it through. Check whether mypy needs the `type: ignore` (chemprop may be untyped); follow how `mypy.ini` treats chemprop.

- [ ] **Step 4: The chemprop engine**

**Manifest:** append `POSITIVE_WEIGHTING, RDKIT_DESCRIPTORS` to `conditions`.

**Descriptor preprocessing**, at module level, numpy imported inside:

```python
def _descriptor_inputs(raw: Any, train_mask: Any) -> tuple[Any, Any]:
    """RDKit descriptors made fit for a neural network, fitted on training rows only.

    A signed log first -- order-preserving and parameter-free -- because RDKit
    descriptors are heavy-tailed (`Ipc` reaches 1e39) and a plain standardization would
    let one outlier squash every other molecule to zero. Then each missing value is
    filled with that descriptor's training-row mean (0.0 for one missing in every
    training row). Standardization is the third step and is chemprop's own:
    `normalize_inputs("X_d")` on the training set, installed as the model's
    `X_d_transform`. Returns the filled matrix and the fill values; predict reapplies
    both from the checkpoint.
    """
    import numpy as np

    logged = np.sign(raw) * np.log1p(np.abs(raw))
    train = logged[train_mask]
    observed = ~np.isnan(train)
    counts = observed.sum(axis=0)
    sums = np.where(observed, train, 0.0).sum(axis=0)
    fill = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return np.where(np.isnan(logged), fill, logged), fill
```

**`_datapoints`** gains `x_d: Sequence[Any] | None = None` and passes `x_d=row` per datapoint.

**`_build_model`** gains:
- `criterion: Any | None`, passed to `BinaryClassificationFFN(n_tasks=…, input_dim=…, criterion=criterion)`;
- `n_descriptors: int`;
- `x_d_transform: Any | None`, passed to `MPNN(..., X_d_transform=x_d_transform)`.

The predictor's `input_dim` becomes `message_passing.output_dim + n_descriptors`, on both the `none` and CheMeleon branches.

**`train`:**

```python
        use_descriptors = bool(conditions["rdkit_descriptors"])
        x_d_all = fill = None
        if use_descriptors:
            from daikonstudio.infrastructure.chem.featurize import rdkit_descriptors

            raw = rdkit_descriptors(ctx.frame[ctx.structure_column].to_list())
            x_d_all, fill = _descriptor_inputs(raw, (ctx.frame["split"] == "train").to_numpy())

        def x_d_for(rows_mask):  # rows of ctx.frame, in frame order
            return None if x_d_all is None else x_d_all[rows_mask]
```

Index the partitions consistently with `train_rows`, `validation_rows` and `test_rows`. Each is `ctx.frame.filter(...)`, which keeps frame order, so `x_d_all[(ctx.frame["split"] == "train").to_numpy()]` lines up.

Then:
- Pass `x_d` to `_datapoints` for the train, validation and test datasets, and in the `score` closure.
- Scaling:

```python
        x_d_transform = None
        if use_descriptors:
            from chemprop.nn.transforms import ScaleTransform

            # Scale the TRAINING set only. ScaleTransform is a no-op in train mode and
            # standardizes in eval mode, and Lightning validates in eval mode -- so
            # normalizing the validation set too (as chemprop's own CLI does) would
            # standardize it twice. Validation, test and predict inputs stay raw (signed
            # log + fill) and the model scales them once.
            x_d_transform = ScaleTransform.from_standard_scaler(train_set.normalize_inputs("X_d"))
```

- Weighting, binary only:

```python
        criterion = None
        mode = str(conditions["positive_weighting"])
        if is_classification and mode != "none":
            from daikonstudio.infrastructure.engines._chemprop_loss import PositiveWeightedBCELoss

            weights = [
                positive_weight(train_rows[column].to_numpy(), mode) or 1.0 for column in columns
            ]
            criterion = PositiveWeightedBCELoss(weights)
```

- Pass `criterion`, `n_descriptors` (`x_d_all.shape[1]` or 0) and `x_d_transform` to `_build_model`.

**Cutoffs.** In `score(rows)`, return probabilities as well. Restructure so:
1. validation values are computed once;
2. cutoffs are picked per column when `ctx.tune_cutoffs and is_classification`, via `mcc_cutoff(validation_rows[column].to_numpy(), values[:, index])`;
3. both partitions are scored with labels `values[:, index] >= (cutoff if cutoff is not None else 0.5)`.

Return `TrainResult(..., cutoffs=cutoffs or None)`. With no tuning the labels are `>= 0.5`, identical to today.

**Checkpoint extras.** After `trainer.save_checkpoint(checkpoint)`:

```python
            if use_descriptors:
                from daikonstudio.infrastructure.chem.featurize import DESCRIPTOR_NAMES

                stored = torch.load(checkpoint, weights_only=False)
                # Read back by `predict`; Lightning ignores keys it does not know on load.
                stored["daikon_descriptors"] = {
                    "names": list(DESCRIPTOR_NAMES),
                    "fill": [float(v) for v in fill],
                }
                torch.save(stored, checkpoint)
```

**`predict`:**
- Read `extras = torch.load(io.BytesIO(ctx.artifact), weights_only=False).get("daikon_descriptors")`. This is safe for the reason `_scoring._load_bundle` gives: the artifact is only ever our own.
- If present:
  - compare `extras["names"]` with `DESCRIPTOR_NAMES`. On a mismatch, raise `ValidationError` with the same wording as `_scoring._require_matching_features`: "This model was trained on a different RDKit descriptor set (… descriptors) from the one this runner computes (…), probably because RDKit was upgraded. Retrain the protocol before predicting.";
  - compute `rdkit_descriptors` for `ctx.frame`, then the signed log, then fill with `np.array(extras["fill"])`, and pass `x_d` to `_datapoints`.
- Checkpoints without the key predict exactly as today.

- [ ] **Step 5: Run tests and gates, then commit**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_chemprop_dmpnn.py -q -rs && uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src && uv run lint-imports
```

Expected: PASS with none skipped. The existing chemprop tests, including legacy-checkpoint ones, pass unchanged.

```bash
git add backend/src/daikonstudio/infrastructure/engines/_chemprop_loss.py backend/src/daikonstudio/infrastructure/engines/chemprop_dmpnn.py backend/tests/unit/engines/test_chemprop_dmpnn.py
git commit -m "feat(engines): chemprop weights positives, takes RDKit descriptors and tunes cutoffs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: MoLFormer-XL weighting and cutoffs

**Files:**
- Modify: `backend/src/daikonstudio/infrastructure/engines/molformer_xl.py`
- Test: `backend/tests/unit/engines/test_molformer_xl.py`

- [ ] **Step 1: Failing test**

A two-target binary frame:
- `epochs=1`, `positive_weighting="balanced"`, `tune_cutoffs=True`;
- assert `result.cutoffs` is a dict (possibly empty, so `None` is allowed when validation is too small); size the frame as in Task 3 so it isn't;
- predict returns `2 × rows`.

A second assertion: `_build_module(model=…, learning_rate=…, is_classification=True, pos_weight=[3.0, 9.0])._loss.pos_weight` equals `torch.tensor([3.0, 9.0])`.

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_molformer_xl.py -q -rs
```

Expected: FAIL, not skipped.

- [ ] **Step 3: Implement**

- Manifest: append `POSITIVE_WEIGHTING`.
- `_build_module(..., pos_weight: list[float] | None = None)`. The classification loss becomes `torch.nn.BCEWithLogitsLoss(pos_weight=None if pos_weight is None else torch.tensor(pos_weight))`, which broadcasts over `(batch, n_tasks)`.
- In `train`, compute `pos_weight = [positive_weight(train_rows[c].to_numpy(), mode) or 1.0 for c in columns]` when classification and `mode != "none"`.
- Cutoffs follow the same three-step restructure as chemprop's `score`.
- `TrainResult(..., cutoffs=cutoffs or None)`.

`predict` is unchanged: the loss never runs at inference, and `_build_module` there keeps `pos_weight=None`.

- [ ] **Step 4: Run tests and gates, then commit**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_molformer_xl.py -q -rs && uv run mypy src && uv run lint-imports
```

Expected: PASS, none skipped.

```bash
git add backend/src/daikonstudio/infrastructure/engines/molformer_xl.py backend/tests/unit/engines/test_molformer_xl.py
git commit -m "feat(engines): MoLFormer-XL weights positives and tunes cutoffs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The training option, recording, Scorecard and prediction

**Files:**
- Modify:
  - `backend/src/daikonstudio/domain/catalog/readout.py` (`threshold`)
  - `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/catalog/repository.py` (`_readout_to_dict` / `_readout_from_dict`)
  - `backend/src/daikonstudio/infrastructure/runner/wire.py` (`ReadoutWire.threshold`, defaulted)
  - `backend/src/daikonstudio/interface/routes/protocols.py`:
    - `ReadoutResponse.threshold`;
    - `TrainProtocolBody.tune_cutoffs`;
    - `ScorecardResponse` with `cutoff`, `baseline_cutoff` and `cutoff_note`.
  - `backend/src/daikonstudio/interface/routes/sweeps.py` (`SubmitSweepBody.tune_cutoffs`)
  - `backend/src/daikonstudio/application/execution/train_protocol.py`
  - `backend/src/daikonstudio/application/execution/sweeps.py`
  - `backend/src/daikonstudio/application/execution/build_scorecard.py`
  - `backend/src/daikonstudio/domain/execution/scorecard.py`
  - `backend/src/daikonstudio/application/catalog/get_scorecard.py`
  - `backend/src/daikonstudio/application/execution/predict_with_protocol.py`
- Test:
  - `backend/tests/integration/test_train_protocol.py`
  - `backend/tests/unit/execution/test_scorecard.py`
  - `backend/tests/api/test_runs.py`
  - `backend/tests/unit/runners/test_wire.py`

**Interfaces:**
- Consumes: `TrainContext.tune_cutoffs`, `TrainResult.cutoffs`, `MIN_CUTOFF_CLASS_COUNT` (Task 1).
- Produces:
  - `TrainProtocolCommand.tune_cutoffs` and `SubmitSweepCommand.tune_cutoffs`;
  - `Readout.threshold`;
  - `TargetInputs.cutoff`, `TargetInputs.baseline_cutoff` and `TargetInputs.cutoff_note`, all defaulted `None`, and the same three on `Scorecard` and `ScorecardResponse`;
  - `primary_metric_ci(..., cutoff=0.5)`;
  - HTTP `tune_cutoffs` on `POST /protocols` and `POST /sweeps`; `ReadoutResponse.threshold`.

- [ ] **Step 1: Failing tests**

In `tests/integration/test_train_protocol.py`, add a binary dataset large enough to tune:
- **The CSV:** about 400 rows of distinct SMILES, alcohols inactive and amines active at about 25%. That leaves a random-split validation of about 40 rows, with at least 10 of each class.
- **A `Studio.dataset(...)` call** with `csv=…` and the binary target spec.
- **Tests:**

```python
async def test_tuned_cutoffs_are_recorded_and_the_baseline_is_tuned_too(studio):
    dataset = await studio.dataset(targets=(TargetSpec(column="y", kind=TargetKind.BINARY),), csv=_tunable_csv())
    run = await studio.wait(
        await studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={}, tune_cutoffs=True)
    )
    assert run.status is RunStatus.READY, run.error_message
    protocol = await studio.protocol_for(run)
    (class_readout,) = [r for r in protocol.readouts if r.type is ReadoutType.CLASS]
    inputs = await studio.scorecard_for(run)
    target = inputs.targets[0]
    assert class_readout.threshold is not None
    assert target.cutoff == class_readout.threshold
    assert target.baseline_cutoff is not None
    assert run.params["tune_cutoffs"] is True


async def test_a_validation_set_too_small_to_tune_keeps_half_and_says_why(studio):
    dataset = await studio.dataset(
        targets=(TargetSpec(column="y", kind=TargetKind.BINARY),), csv=_csv(_alternating_values())
    )  # the existing 20-compound fixture: validation of 2 rows
    run = await studio.wait(
        await studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={}, tune_cutoffs=True)
    )
    target = (await studio.scorecard_for(run)).targets[0]
    assert target.cutoff is None
    assert "at least 10" in (target.cutoff_note or "")
```

`Studio.train` gains a `tune_cutoffs: bool = False` kwarg, passed into the command. That file's `Studio` has no `protocol_for`; add one, copied from `tests/integration/test_prediction_cache.py:178`. In `tests/api/test_runs.py`, train with `tune_cutoffs=True` on the tunable CSV and publish. Predict, then assert every row's class equals `probability >= readout["threshold"]`, reading both from `/runs/{id}/results` and the protocol. In `tests/unit/execution/test_scorecard.py`, assert `primary_metric_ci(..., cutoff=0.9)` differs from the 0.5 result on a constructed example. In `test_wire.py`, round-trip a `Readout` with `threshold=0.3`, and one without.

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/integration/test_train_protocol.py -k "tuned or too_small" tests/unit/execution/test_scorecard.py tests/unit/runners/test_wire.py -q
```

Expected: FAIL.

- [ ] **Step 3: Recording the cutoff on the readout**

**`domain/catalog/readout.py`:** `Readout` gains:

```python
    # The probability at or above which a CLASS readout reads 1. None means 0.5 -- every
    # protocol trained before cutoffs could be tuned, and every untuned one since.
    threshold: float | None = None
```

**Catalog repository:**
- `_readout_to_dict` writes `"threshold": readout.threshold`;
- `_readout_from_dict` reads `data.get("threshold")`. Old rows lack the key.

**Wire:** `ReadoutWire.threshold: float | None = None`, mapped both ways.

**Routes:** `ReadoutResponse.threshold: float | None`, mapped.

- [ ] **Step 4: The training option**

**`TrainProtocolCommand`:**
- gains `tune_cutoffs: bool = False`;
- `to_params` writes `"tune_cutoffs": self.tune_cutoffs`;
- `from_params` reads `params.get("tune_cutoffs", False)`.

**`TrainProtocol.__call__`:** add `tune_cutoffs=command.tune_cutoffs` to `compute_cache_key(...)`.

**`SubmitSweepCommand`:** gains `tune_cutoffs: bool = False`, passed into each child `TrainProtocolCommand`.

**HTTP:** `TrainProtocolBody.tune_cutoffs: bool = False` and `SubmitSweepBody.tune_cutoffs: bool = False`, passed through by both routes.

**`RunTraining`.** At the start of `__call__`, next to `self._deadline_at`:

```python
        # The same request reaches all three fits: a tuned model measured against an
        # untuned baseline would flatter the model by construction.
        self._tune_cutoffs = command.tune_cutoffs
```

This happens after `command` is parsed. Initialize `self._tune_cutoffs = False` in `__init__`. `_train_off_thread` passes `tune_cutoffs=self._tune_cutoffs` to `TrainContext`.

Per target, while building `TargetInputs`:

```python
                    cutoff=(chosen.cutoffs or {}).get(target.column),
                    baseline_cutoff=(baseline_result.cutoffs or {}).get(target.column),
                    cutoff_note=_cutoff_note(target, task, frame) if self._tune_cutoffs else None,
```

`_cutoff_note(target, task, frame) -> str | None` returns `None` for regression or when the chosen cutoff exists. Otherwise:
- **No validation rows:** "Not tuned: this split has no validation set, so the cutoff stays at 0.5."
- **Too few of a class:** "Not tuned: the validation set has {positives} active and {negatives} inactive compounds for '{column}'; at least {MIN_CUTOFF_CLASS_COUNT} of each are needed, so the cutoff stays at 0.5." Count from `frame.filter(pl.col("split") == "validation")[target.column]`.
- **Otherwise:** "Not tuned: the model's validation predictions could not support a cutoff, so it stays at 0.5."

Pass the chosen cutoff in, so the helper only explains a missing one.

Readouts:

```python
        readouts = tuple(
            replace(readout, threshold=(chosen.cutoffs or {}).get(readout.name))
            if readout.type is ReadoutType.CLASS
            else readout
            for readout in derive_readouts(dataset.targets)
        )
```

Pass `readouts=readouts` to `InSilicoProtocol`. A CLASS readout's name is its target column.

**`TargetInputs`:** gains `cutoff: float | None = None`, `baseline_cutoff: float | None = None` and `cutoff_note: str | None = None`. Defaults keep older blobs readable.

- [ ] **Step 5: Scorecard**

**`build_scorecard.primary_metric_ci`:** gains `cutoff: float = 0.5`, and uses `p[idx] >= cutoff` in the MCC branch. Its docstring changes "MCC at the 0.5 threshold" to "MCC at the model's decision cutoff".

**`build_scorecard`:**
- gains `cutoff: float | None = None`, `baseline_cutoff: float | None = None` and `cutoff_note: str | None = None`;
- passes `cutoff=cutoff if cutoff is not None else 0.5` to `primary_metric_ci`;
- sets the three new `Scorecard` fields.

**`Scorecard`:** gains the three fields, documented as "the decision cutoffs MCC and balanced accuracy were measured at; None means 0.5".

**`get_scorecard._build_all`:** passes `target.cutoff`, `target.baseline_cutoff` and `target.cutoff_note`.

**`ScorecardResponse`:** gains the three fields.

- [ ] **Step 6: Prediction**

In `RunPrediction`, the CLASS branch becomes:

```python
            if readouts[column].type is ReadoutType.CLASS:
                cutoff = readouts[column].threshold
                cutoff = 0.5 if cutoff is None else cutoff
                # (keep the explanation that value is P(class=1) and the class is derived
                # here; replace the fixed-0.5 ponytail note with: "the cutoff is 0.5 unless
                # training tuned it on validation (`tune_cutoffs`), when it is the one stored
                # on the readout.")
                columns[probability_column(column)] = pl.Series(values, dtype=pl.Float64)
                columns[column] = pl.Series(
                    [1.0 if v >= cutoff else 0.0 for v in values], dtype=pl.Float64
                )
```

- [ ] **Step 7: Run tests and gates, then commit**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/integration/test_train_protocol.py tests/unit/execution tests/unit/catalog tests/unit/runners tests/api/test_runs.py tests/api/test_protocols.py -q && OMP_NUM_THREADS=1 uv run pytest tests/integration/test_sweeps.py -q && uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src && uv run lint-imports
```

Expected: PASS.

```bash
git add backend/src/daikonstudio backend/tests
git commit -m "feat(training): tune decision cutoffs on validation for the model and its baseline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Frontend

**Files:**
- Modify: `frontend/openapi.json` and `frontend/src/shared/lib/api/**` (run `make generate-api`)
- Modify: `frontend/src/features/protocols/components/condition-fields.tsx` (hide by `tasks`)
- Modify:
  - `frontend/src/features/protocols/components/train-protocol-form.tsx`
  - `frontend/src/features/sweeps/components/sweep-form.tsx`
  - `frontend/src/features/sweeps/hooks/use-sweeps.ts` (body type)
- Modify:
  - `frontend/src/features/protocols/components/scorecard-view.tsx` (the cutoff line in the verdict band)
  - `frontend/src/features/protocols/components/protocol-detail.tsx` (readout threshold)
- Test: the existing component tests beside these files, plus `condition-fields` (create `condition-fields.test.tsx` if absent).

**Interfaces:**
- Consumes:
  - `ConditionResponse.tasks`;
  - `ReadoutResponse.threshold`;
  - `ScorecardResponse.cutoff`, `.baseline_cutoff` and `.cutoff_note`;
  - `tune_cutoffs` on the training and sweep bodies.

- [ ] **Step 1: Regenerate**

Run `make generate-api`. The diff must contain only:
- `tasks` on conditions;
- `threshold` on readouts;
- the three scorecard fields;
- `tune_cutoffs` on both bodies.

- [ ] **Step 2: Failing tests**

**`condition-fields`:**

```tsx
it("hides a setting that applies only to tasks the dataset does not have", () => {
  render(
    <ConditionFields
      conditions={[
        { key: "positive_weighting", label: "Positive-class weighting", type: "enum", required: false,
          default: "none", minimum: null, maximum: null, options: ["none", "balanced"], help: null,
          tasks: ["binary_classification"] },
        { key: "epochs", label: "Training epochs", type: "integer", required: false, default: 50,
          minimum: 1, maximum: 500, options: [], help: null, tasks: [] },
      ]}
      tasks={["regression"]}
      /* plus this component's existing required props (values, onChange, ...) */
    />,
  );
  expect(screen.queryByText("Positive-class weighting")).not.toBeInTheDocument();
  expect(screen.getByText("Training epochs")).toBeInTheDocument();
});
```

**Train form:**
- with a binary dataset, a "Tune decision cutoffs" checkbox exists, and submitting with it checked sends `tune_cutoffs: true`;
- with a numeric-only dataset, the checkbox is absent.

Follow the file's mock patterns.

**Scorecard view:** a scorecard with `cutoff: 0.031` and `baseline_cutoff: 0.12` renders "cutoff 0.031".

- [ ] **Step 3: Implement**

**`ConditionFields`:**
- optional prop `tasks?: string[]`;
- render only conditions where `!condition.tasks?.length || !tasks || condition.tasks.some((task) => tasks.includes(task))`.

**Train form and sweep form:**
- Derive the dataset's tasks:
  ```ts
  const tasks = [...new Set(dataset.targets.map((t) => TASK_FOR_TARGET_KIND[t.kind]))]
  ```
- Pass `tasks` to every `ConditionFields`, for the engine and the baseline.
- When `tasks.includes("binary_classification")`, render a checkbox, styled like other boolean inputs in the form:
  - label: "Tune decision cutoffs"
  - help: "Chooses each active/inactive label's cutoff to maximize MCC on the validation set. Applies to the model and its baseline alike."
- State `tuneCutoffs` defaults to false. Send `tune_cutoffs: tuneCutoffs` in the body.
- In the sweep form, add `tune_cutoffs?: boolean` to the `useSubmitSweep` body type.

**Scorecard `VerdictBand`:** under the MCC line, when `scorecard.cutoff != null`, render a muted line:

"At cutoff {cutoff.toPrecision(2)}, tuned on validation. Baseline at its own tuned cutoff {baseline_cutoff.toPrecision(2)}."

If `baseline_cutoff` is null, end the line after "tuned on validation." instead. When `cutoff_note`, render the note instead.

**Protocol readouts card:** for a CLASS readout with `threshold != null`, append the muted text "class at cutoff {threshold.toPrecision(2)}".

- [ ] **Step 4: Gates, then commit**

Run:
```
cd frontend && pnpm vitest run src/features/protocols src/features/sweeps && pnpm lint && pnpm exec tsc --noEmit
```

Expected: PASS.

```bash
git add frontend
git commit -m "feat(frontend): tune-cutoffs option, task-scoped settings, cutoffs on the scorecard

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Verify (controller)

- [ ] **Step 1: Run the full suites once, in CI order**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest -q
cd ../frontend && pnpm test
```

- [ ] **Step 2: Restart the runners**

Run `make dev-worker && make dev-worker-gpu`. The API and frontend reload on their own.

- [ ] **Step 3: Real data, through the UI**

On `nuisance_sample_10k`, start one sweep with "Tune decision cutoffs" on and four chemprop configurations:

| # | Pretrained | Weighting | Descriptors |
|---|---|---|---|
| 1 | none | Off | Off |
| 2 | none | Balanced | On |
| 3 | CheMeleon | Off | Off |
| 4 | CheMeleon | Balanced | On |

Record per-label ROC-AUC, PR-AUC and MCC (with the cutoff) from each protocol's scorecard. Put them beside the paper's graph-only row (ROC-AUC 0.941, PR-AUC 0.686, MCC 0.615, macro-averaged).

- [ ] **Step 4: Final audit**

Run one whole-branch audit (most capable model), then one fix wave and one scoped re-review.
