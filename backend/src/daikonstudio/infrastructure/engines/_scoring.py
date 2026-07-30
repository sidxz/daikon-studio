"""Metrics and ensemble-uncertainty prediction shared by both ECFP4 engines.

Plain accuracy is never computed here -- not even as an unused local. A dataset
that is 99.9% negative yields a 99.9%-accurate model that predicts nothing
useful, and the only reliable way to keep that number off a Scorecard is to
never calculate it in the first place.
"""

from __future__ import annotations

import pickle
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

from daikonstudio.application.engines.context import PredictContext, TrainContext
from daikonstudio.infrastructure.chem.featurize import ecfp4

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


def _score(
    model: Any, test_rows: pl.DataFrame, ctx: TrainContext, is_classification: bool
) -> dict[str, float]:
    """RMSE/MAE/R2 for regression; MCC/balanced accuracy/AUROC/AUPRC for classification."""
    x_test = ecfp4(test_rows[ctx.structure_column].to_list())
    y_test = test_rows[ctx.target_column].to_numpy()

    if not is_classification:
        return regression_metrics(y_test, model.predict(x_test))

    # Both single-class checks stay HERE, before any sklearn call -- not delegated to
    # `classification_metrics` -- because short-circuiting is what keeps sklearn's
    # "y_pred contains classes not in y_true" warning from firing at all.
    if len(np.unique(y_test)) < 2 or len(model.classes_) < 2:
        return _undefined_classification_metrics()

    return classification_metrics(
        y_test,
        model.predict(x_test),
        _positive_class_probability(model, x_test),
        train_has_both_classes=True,
    )


def _predict_with_tree_ensemble(ctx: PredictContext) -> pl.DataFrame:
    """Returns row_id (int), value (float), uncertainty (float | null).

    RandomForest exposes its individual trees via `estimators_`; XGBoost's sklearn
    wrapper does not, so that attribute is used to tell the two apart rather than
    threading an extra "which engine" flag through the artifact.
    """
    # pickle.loads executes arbitrary code for a crafted payload. Safe here:
    # `ctx.artifact` is never user-supplied bytes -- it is produced exclusively by
    # this engine's own `train()` and round-tripped through our own blob storage,
    # never accepted from an external upload.
    bundle: dict[str, Any] = pickle.loads(ctx.artifact)
    model: Any = bundle["model"]
    is_classification: bool = bundle["is_classification"]

    x = ecfp4(ctx.frame[ctx.structure_column].to_list())
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

    # Explicit dtypes, not inferred: an all-None `uncertainty` list (the XGBoost case)
    # infers as polars' Null dtype rather than a nullable Float64, which would make
    # the two engines' predict() outputs schema-incompatible for a caller that
    # concatenates or persists results across engines.
    return pl.DataFrame(
        {
            "row_id": pl.Series(row_ids, dtype=pl.Int64),
            "value": pl.Series([float(v) for v in value], dtype=pl.Float64),
            "uncertainty": pl.Series(uncertainty, dtype=pl.Float64),
        }
    )
