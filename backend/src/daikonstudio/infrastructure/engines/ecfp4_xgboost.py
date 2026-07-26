"""ECFP4 + XGBoost. The gradient-boosted alternative to the RandomForest baseline."""

from __future__ import annotations

import pickle

import polars as pl
from xgboost import XGBClassifier, XGBRegressor

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
    id="ecfp4-xgboost",
    version="1.0.0",
    name="ECFP4 + XGBoost",
    description="Morgan fingerprints with gradient-boosted trees. Often sharper than "
    "the random forest baseline, at the cost of being more sensitive to its settings.",
    tasks=(TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION),
    conditions=(
        ConditionSpec(
            key="n_estimators",
            label="Number of boosting rounds",
            type=ConditionType.INTEGER,
            default=400,
            minimum=10,
            maximum=2000,
            help="How many trees are added one after another. More rounds can fit "
            "the data more closely but risk overfitting past a point.",
        ),
        ConditionSpec(
            key="max_depth",
            label="Maximum tree depth",
            type=ConditionType.INTEGER,
            default=6,
            minimum=1,
            maximum=20,
            help="How many splits deep each tree can grow. Shallower trees generalize "
            "better; deeper trees can capture more complex structure-activity patterns.",
        ),
        ConditionSpec(
            key="learning_rate",
            label="Learning rate",
            type=ConditionType.NUMBER,
            default=0.1,
            minimum=0.001,
            maximum=1.0,
            help="How much each boosting round is allowed to correct the last. Lower "
            "values need more rounds but usually generalize better.",
        ),
    ),
    is_baseline=False,
)


class Ecfp4XGBoost:
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

        model: XGBClassifier | XGBRegressor
        model_kwargs = {
            "n_estimators": conditions["n_estimators"],
            "max_depth": conditions["max_depth"],
            "learning_rate": conditions["learning_rate"],
            "random_state": ctx.seed,
            # n_jobs=-1: unlike RandomForest's predict() (see ecfp4_randomforest.py),
            # XGBoost's hist tree builder is thread-count deterministic by design --
            # measured bit-identical metrics across five fit+predict runs at n_jobs=-1
            # on the same seed, so there is no reproducibility tradeoff to make here.
            "n_jobs": -1,
        }
        if is_classification:
            model = XGBClassifier(**model_kwargs)
        else:
            model = XGBRegressor(**model_kwargs)
        model.fit(x_train, y_train)

        # pickle.dumps serializes the fitted model; safe to write, since only our
        # own predict() ever reads this artifact back (see _scoring.py for the load
        # side, and its comment on why deserializing it is safe there).
        artifact = pickle.dumps({"model": model, "is_classification": is_classification})
        return TrainResult(
            artifact=artifact,
            metrics=_score(model, test_rows, ctx, is_classification),
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        return _predict_with_tree_ensemble(ctx)
