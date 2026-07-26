"""ECFP4 + RandomForest. The mandatory baseline every Scorecard compares against.

In the Polaris ADMET competition, fingerprint-plus-random-forest baselines placed
around 20th of 66 teams. A scientist whose model cannot beat this needs to know on
the first screen, so this has to be a genuinely competitive model, not a strawman.
"""

from __future__ import annotations

import io

import joblib  # type: ignore[import-untyped]
import polars as pl
from sklearn.ensemble import (  # type: ignore[import-untyped]
    RandomForestClassifier,
    RandomForestRegressor,
)

from daikonstudio.application.engines.context import PredictContext, TrainContext, TrainResult
from daikonstudio.application.engines.manifest import (
    ConditionSpec,
    ConditionType,
    EngineManifest,
    TaskType,
    validate_conditions,
)
from daikonstudio.infrastructure.chem.featurize import ecfp4
from daikonstudio.infrastructure.engines._scoring import _predict_with_tree_ensemble, _score

_MANIFEST = EngineManifest(
    id="ecfp4-randomforest",
    version="1.0.0",
    name="ECFP4 + Random Forest",
    description=(
        "Morgan fingerprints with a random forest. Fast, robust, and the required "
        "baseline every model on this platform is measured against."
    ),
    tasks=(TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION),
    conditions=(
        ConditionSpec(
            key="n_estimators",
            label="Number of trees",
            type=ConditionType.INTEGER,
            default=500,
            minimum=10,
            maximum=2000,
            help="How many decision trees to average over. More trees give steadier "
            "predictions but take longer to train. 500 is a good default.",
        ),
    ),
    is_baseline=True,
)


class Ecfp4RandomForest:
    @staticmethod
    def manifest() -> EngineManifest:
        return _MANIFEST

    def train(self, ctx: TrainContext) -> TrainResult:
        conditions = validate_conditions(_MANIFEST, ctx.conditions)
        train_rows = ctx.frame.filter(pl.col("split") == "train")
        test_rows = ctx.frame.filter(pl.col("split") == "test")

        x_train = ecfp4(train_rows[ctx.structure_column].to_list())
        y_train = train_rows[ctx.target_column].to_numpy()
        is_classification = ctx.task is TaskType.BINARY_CLASSIFICATION

        # No annotation on `model`: sklearn ships no py.typed marker, so both
        # constructors already resolve to Any (see the import above) -- declaring a
        # union of the two here would just be a union of Any, which mypy rejects as
        # a type. Any is what falls out naturally by leaving it unannotated.
        #
        # n_jobs=1, not -1: measured empirically that n_jobs=-1 makes RandomForest's
        # parallel prediction-averaging non-reproducible at the float ULP level --
        # thread completion order varies run to run, and float addition isn't
        # associative, so the same seed can yield RMSE that differs in the 16th
        # digit. That's enough to break exact-equality reproducibility checks.
        if is_classification:
            model = RandomForestClassifier(
                n_estimators=conditions["n_estimators"], random_state=ctx.seed, n_jobs=1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=conditions["n_estimators"], random_state=ctx.seed, n_jobs=1
            )
        model.fit(x_train, y_train)

        buffer = io.BytesIO()
        # joblib.dump pickles the fitted model; safe to write, since only our own
        # predict() ever reads this artifact back (see _scoring.py for the load side).
        joblib.dump({"model": model, "is_classification": is_classification}, buffer)
        return TrainResult(
            artifact=buffer.getvalue(),
            metrics=_score(model, test_rows, ctx, is_classification),
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        return _predict_with_tree_ensemble(ctx)
