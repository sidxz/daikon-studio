"""The shared metric vocabulary, plus ensemble-uncertainty prediction for the tree engines.

`regression_metrics` and `classification_metrics` are engine-agnostic and imported by
every engine, chemprop included: a Scorecard comparing "your model" against "the
baseline" is only meaningful while both numbers come from literally the same code.
`_scored` below is sklearn-shaped and stays private to the sklearn-API engines -- the
three ECFP4 trees, the descriptor one and the Gaussian process. Those differ only in
how a structure becomes a feature matrix, which is why the featurizer is a parameter
here rather than four near-copies of the same scoring code.

The two `_predict_with_*` functions do not collapse the same way: they share everything
except how uncertainty is obtained, and that is exactly what distinguishes the engines
from each other. What they do share -- the artifact bundle's shape and the output
schema -- lives in `_prediction_frame` and the featurizer lookup below.

Plain accuracy is never computed here -- not even as an unused local. A dataset
that is 99.9% negative yields a 99.9%-accurate model that predicts nothing
useful, and the only reliable way to keep that number off a Scorecard is to
never calculate it in the first place.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

from daikonstudio.application.engines.context import (
    MIN_CUTOFF_CLASS_COUNT,
    PredictContext,
    TrainContext,
)
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.chem.featurize import (
    DESCRIPTOR_NAMES,
    ecfp4,
    rdkit_descriptors,
)

#: How a fitted artifact names the representation it was trained on. `predict` reads the
#: name off the bundle rather than taking it as an argument, because nothing at the
#: predict call site knows which engine produced the artifact -- `_predict_with_tree_
#: ensemble` is reached through `Engine.predict`, which only has bytes and a frame.
Featurizer = Callable[[list[str]], np.ndarray]
_FEATURIZERS: dict[str, Featurizer] = {"ecfp4": ecfp4, "rdkit_descriptors": rdkit_descriptors}

#: Only for featurizers whose columns are named and can drift; see
#: `_require_matching_features`. ECFP4's hashed bits are deliberately absent.
_FEATURE_NAMES: dict[str, tuple[str, ...]] = {"rdkit_descriptors": DESCRIPTOR_NAMES}

# `model` is `Any` throughout this module: it is either a scikit-learn estimator
# (no py.typed marker, so mypy already erases it to Any -- see the import above) or
# an xgboost estimator, and the two engines have no shared typed base to name.
# Fighting mypy for a union of untyped sklearn classes bought nothing but a broken
# type alias; Any is what the sklearn side already collapses to.


def _positive_class_probability(model: Any, x: np.ndarray) -> np.ndarray:
    """P(class=1), robust to a model that only ever saw one class while training.

    sklearn's `predict_proba` returns one column per class *the model knows about*,
    not one per class in some absolute label space -- so when training data is
    single-class, the output is shape (n, 1), and there is no column 1 to index.
    That column is mapped back to P(class=1) explicitly instead: it's 1 - column0
    when the model's only known class is 0, or column0 itself when it's 1.
    """
    proba = model.predict_proba(x)
    if proba.shape[1] < 2:
        only_class = model.classes_[0]
        return proba[:, 0] if only_class == 1 else 1.0 - proba[:, 0]  # type: ignore[no-any-return]
    return proba[:, 1]  # type: ignore[no-any-return]


def _undefined_classification_metrics() -> dict[str, float]:
    """All four together, uniformly. MCC has no defined value on a single-class split,
    and balanced accuracy silently collapses to plain accuracy -- reporting three of
    four would imply the missing one was the only problem."""
    return {
        "mcc": float("nan"),
        "balanced_accuracy": float("nan"),
        "auroc": float("nan"),
        "auprc": float("nan"),
    }


def regression_metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """RMSE, MAE and R2 -- the regression half of the shared vocabulary.

    Engine-agnostic on purpose: this is the code a chemprop model and the ECFP4
    baseline are both measured by, which is what makes a Scorecard's comparison mean
    anything.
    """
    return {
        "rmse": float(root_mean_squared_error(y_true, predicted)),
        "mae": float(mean_absolute_error(y_true, predicted)),
        "r2": float(r2_score(y_true, predicted)),
    }


def classification_metrics(
    y_true: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    train_has_both_classes: bool,
) -> dict[str, float]:
    """MCC, balanced accuracy, AUROC and AUPRC. Never plain accuracy.

    `labels` are hard 0/1 predictions and `probabilities` is P(class=1); both are passed
    rather than derived, because sklearn's `predict` and a 0.5 threshold on
    `predict_proba` are the same thing for these estimators and an engine that only has
    probabilities (chemprop) should threshold them explicitly rather than have this
    function guess.

    `train_has_both_classes` generalises what used to be a `model.classes_` check, so an
    engine with no such attribute can answer the same question.
    """
    if len(np.unique(y_true)) < 2 or not train_has_both_classes:
        return _undefined_classification_metrics()
    return {
        "mcc": float(matthews_corrcoef(y_true, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, labels)),
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
    }


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

    # Both single-class checks stay HERE, before any sklearn call -- not delegated to
    # `classification_metrics` -- because short-circuiting is what keeps sklearn's
    # "y_pred contains classes not in y_true" warning from firing at all.
    if len(np.unique(y)) < 2 or len(model.classes_) < 2:
        return _undefined_classification_metrics()

    probabilities = _positive_class_probability(model, x)
    labels = model.predict(x) if cutoff is None else (probabilities >= cutoff).astype(int)
    return classification_metrics(y, labels, probabilities, train_has_both_classes=True)


def _scored(
    model: Any, ctx: TrainContext, is_classification: bool, featurizer: Featurizer = ecfp4
) -> tuple[
    dict[str, dict[str, float]], dict[str, dict[str, float]] | None, dict[str, float] | None
]:
    """Test metrics, validation metrics and the tuned cutoff for the one target this fit
    trained on, keyed by its column -- the shapes `TrainResult` takes now that a Dataset
    can hold several. These engines always fit one target; `FanOut` merges the keys.

    `featurizer` defaults to `ecfp4` so the ECFP4 engines read unchanged; the descriptor
    engine passes its own. It must be the same one `train` fitted on -- scoring a model
    against a different representation than it learned produces numbers rather than an
    error.

    Validation is scored with the identical code as test, on purpose: a validation number
    a scientist is asked to tune against has to be the same measurement as the one they
    will eventually be judged by, or tuning against it optimizes the wrong thing. It is
    `None` when the partition is empty, which a split with a zero validation fraction
    produces legitimately.

    Validation is also where the cutoff comes from when `ctx.tune_cutoffs` is on, and test
    is then scored at that cutoff. Validation metrics at a tuned cutoff are optimistic,
    since the cutoff was chosen on them -- the test numbers stay the verdict. Each
    partition is featurized once.
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


_FEATURIZERS["ecfp4+rdkit_descriptors"] = _ecfp4_with_descriptors
_FEATURE_NAMES["ecfp4+rdkit_descriptors"] = DESCRIPTOR_NAMES


def tree_featurizer(conditions: dict[str, Any]) -> tuple[str, Featurizer]:
    """The bundle key and featurizer an ECFP4 tree engine fits with, from its settings."""
    key = "ecfp4+rdkit_descriptors" if conditions.get("rdkit_descriptors") else "ecfp4"
    return key, _FEATURIZERS[key]


def bundle_features(featurizer_key: str) -> dict[str, Any]:
    """What a tree bundle records about its input, so predict featurizes identically and
    `_require_matching_features` can refuse a descriptor list that has since changed.

    Nothing for plain ECFP4: a default fit's bundle stays exactly what it was before this
    existed, and `_load_bundle` reads a bundle with no key as ECFP4, as it always has.
    """
    if featurizer_key == "ecfp4":
        return {}
    names = _FEATURE_NAMES.get(featurizer_key)
    return {"featurizer": featurizer_key} | ({"feature_names": names} if names else {})


def _require_matching_features(bundle: dict[str, Any]) -> None:
    """Refuse to predict through a featurizer that has changed shape since the fit.

    `Descriptors.descList` is a property of the installed RDKit, not a constant --
    RDKit adds descriptors between releases. Upgrading it under a stored model shifts
    every column by one, which no estimator can detect: XGBoost sees the right number
    of floats and returns confident nonsense. This is the one failure mode in the
    predict path that is silent, so it is the one worth an explicit check.

    Artifacts written before this key existed carry no names and are not checked; they
    are ECFP4 models, whose 2048 hashed bits have no names to drift.
    """
    stored = bundle.get("feature_names")
    if stored is None:
        return
    current = _FEATURE_NAMES.get(bundle.get("featurizer", "ecfp4"))
    if current is None or tuple(stored) == tuple(current):
        return
    raise ValidationError(
        "This model was trained on a different RDKit descriptor set "
        f"({len(stored)} descriptors) from the one this runner computes "
        f"({len(current)}), probably because RDKit was upgraded. Retrain the protocol "
        "before predicting."
    )


def _load_bundle(ctx: PredictContext) -> tuple[dict[str, Any], np.ndarray]:
    """The artifact bundle and the feature matrix it expects, from bytes alone.

    Shared by both predict paths because both read the same bundle shape; they diverge
    only after this, on where uncertainty comes from.
    """
    # pickle.loads executes arbitrary code for a crafted payload. Safe here:
    # `ctx.artifact` is never user-supplied bytes -- it is produced exclusively by
    # an engine's own `train()` and round-tripped through our own blob storage,
    # never accepted from an external upload.
    bundle: dict[str, Any] = pickle.loads(ctx.artifact)

    # Default, not `bundle["featurizer"]`: every artifact written before the descriptor
    # engine existed is an ECFP4 one and has no such key. Those models still predict.
    featurizer = _FEATURIZERS[bundle.get("featurizer", "ecfp4")]
    _require_matching_features(bundle)
    return bundle, featurizer(ctx.frame[ctx.structure_column].to_list())


def _prediction_frame(value: np.ndarray, uncertainty: Sequence[float | None]) -> pl.DataFrame:
    """row_id (int), value (float), uncertainty (float | null) -- the `predict` contract.

    Explicit dtypes, not inferred: an all-None `uncertainty` (the XGBoost case) infers
    as polars' Null dtype rather than a nullable Float64, which would make two engines'
    `predict()` outputs schema-incompatible for a caller that concatenates or persists
    results across engines. One function so that invariant holds for every engine
    rather than for whichever ones remembered it.
    """
    return pl.DataFrame(
        {
            "row_id": pl.Series(list(range(len(value))), dtype=pl.Int64),
            "value": pl.Series([float(v) for v in value], dtype=pl.Float64),
            "uncertainty": pl.Series(list(uncertainty), dtype=pl.Float64),
        }
    )


def _predict_with_tree_ensemble(ctx: PredictContext) -> pl.DataFrame:
    """Returns row_id (int), value (float), uncertainty (float | null).

    RandomForest exposes its individual trees via `estimators_`; XGBoost's sklearn
    wrapper does not, so that attribute is used to tell the two apart rather than
    threading an extra "which engine" flag through the artifact.
    """
    bundle, x = _load_bundle(ctx)
    model: Any = bundle["model"]
    is_classification: bool = bundle["is_classification"]

    row_ids = list(range(x.shape[0]))
    has_ensemble_spread: bool = hasattr(model, "estimators_")
    uncertainty: list[float | None]

    if is_classification:
        value = _positive_class_probability(model, x)
        if has_ensemble_spread:
            # Distance from the decision boundary: 0.5 (a coin flip) is maximally
            # uncertain, 0.0/1.0 is maximally certain.
            uncertainty = [float(v) for v in (1.0 - 2.0 * np.abs(value - 0.5))]
        else:
            # XGBoost has no ensemble spread to report; a fabricated number would be
            # plotted as if it meant something, which is worse than an absent one.
            uncertainty = [None] * len(row_ids)
    else:
        value = model.predict(x)
        if has_ensemble_spread:
            tree_predictions = np.stack([tree.predict(x) for tree in model.estimators_])
            uncertainty = [float(v) for v in tree_predictions.std(axis=0)]
        else:
            uncertainty = [None] * len(row_ids)

    return _prediction_frame(value, uncertainty)


def _predict_with_gaussian_process(ctx: PredictContext) -> pl.DataFrame:
    """Same contract as `_predict_with_tree_ensemble`, different source of uncertainty.

    Separate rather than another branch inside that function, because a GP's uncertainty
    is not a proxy derived from its outputs -- it is a second thing the model returns.
    Routing a GP through the tree path would find no `estimators_` and report `None`,
    throwing away the one property the engine exists for.
    """
    bundle, x = _load_bundle(ctx)
    model: Any = bundle["model"]

    if bundle["is_classification"]:
        # GaussianProcessClassifier's Laplace approximation exposes no latent variance,
        # so this is the same distance-from-the-boundary the forest reports for a class
        # probability: 0.5 is a coin flip, 0.0/1.0 is certain. Honest, but a different
        # quantity from the regression branch's posterior std -- which is the roster's
        # existing "uncertainty means several things" problem, not a new one.
        value = _positive_class_probability(model, x)
        spread = 1.0 - 2.0 * np.abs(value - 0.5)
    else:
        # `normalize_y=True` at fit time means sklearn un-scales both of these, so the
        # standard deviation comes back in the target's own units.
        value, spread = model.predict(x, return_std=True)

    return _prediction_frame(value, [float(v) for v in spread])
