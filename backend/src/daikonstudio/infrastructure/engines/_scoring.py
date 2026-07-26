"""Metrics and ensemble-uncertainty prediction shared by both ECFP4 engines.

Plain accuracy is never computed here -- not even as an unused local. A dataset
that is 99.9% negative yields a 99.9%-accurate model that predicts nothing
useful, and the only reliable way to keep that number off a Scorecard is to
never calculate it in the first place.
"""

from __future__ import annotations

import io
from typing import Any

import joblib  # type: ignore[import-untyped]
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


def _score(
    model: Any, test_rows: pl.DataFrame, ctx: TrainContext, is_classification: bool
) -> dict[str, float]:
    """RMSE/MAE/R2 for regression; MCC/balanced accuracy/AUROC/AUPRC for classification."""
    x_test = ecfp4(test_rows[ctx.structure_column].to_list())
    y_test = test_rows[ctx.target_column].to_numpy()

    if not is_classification:
        predictions = model.predict(x_test)
        return {
            "rmse": float(root_mean_squared_error(y_test, predictions)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
        }

    predictions = model.predict(x_test)
    metrics = {
        "mcc": float(matthews_corrcoef(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
    }
    if len(np.unique(y_test)) < 2:
        # AUROC/AUPRC are undefined with only one class present -- a tiny test split
        # (or an unlucky one) can land here. NaN says "undefined", not "bad model".
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    else:
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics["auroc"] = float(roc_auc_score(y_test, probabilities))
        metrics["auprc"] = float(average_precision_score(y_test, probabilities))
    return metrics


def _predict_with_tree_ensemble(ctx: PredictContext) -> pl.DataFrame:
    """Returns row_id (int), value (float), uncertainty (float | null).

    RandomForest exposes its individual trees via `estimators_`; XGBoost's sklearn
    wrapper does not, so that attribute is used to tell the two apart rather than
    threading an extra "which engine" flag through the artifact.
    """
    # joblib.load deserializes a pickle, which can execute arbitrary code for a
    # crafted payload. Safe here: `ctx.artifact` is never user-supplied bytes -- it
    # is produced exclusively by this engine's own `train()` and round-tripped
    # through our own blob storage, never accepted from an external upload.
    bundle: dict[str, Any] = joblib.load(io.BytesIO(ctx.artifact))
    model: Any = bundle["model"]
    is_classification: bool = bundle["is_classification"]

    x = ecfp4(ctx.frame[ctx.structure_column].to_list())
    row_ids = list(range(x.shape[0]))
    has_ensemble_spread: bool = hasattr(model, "estimators_")
    uncertainty: list[float | None]

    if is_classification:
        value = model.predict_proba(x)[:, 1]
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
