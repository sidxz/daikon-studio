"""RDKit 2D descriptors + XGBoost -- the recipe that actually wins ADMET benchmarks.

Every entry that survived the February 2026 reproducibility audit of the TDC ADMET
leaderboard (CaliciBoost, MapLight, MapLight+GNN) is descriptors plus boosting, and
across matched-budget comparisons rich descriptors beat hashed fingerprints by a wide
margin -- 56.9% win rate for RDKit2D against Morgan's 22.4% over 58 tasks. Until this
engine, nothing in the roster could express that: every engine saw either 2048 hashed
Morgan bits or the raw molecular graph.

Boosting rather than a random forest, for a specific reason. The featurizer emits NaN
for anything unmeasurable (see `rdkit_descriptors`), XGBoost treats NaN as missing and
learns a split direction for it, and sklearn's RandomForest raises on NaN outright. The
representation and the estimator are chosen together here.

Runs on the default lane: RDKit plus XGBoost, no torch, no GPU.
"""

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
from daikonstudio.infrastructure.chem.featurize import DESCRIPTOR_NAMES, rdkit_descriptors
from daikonstudio.infrastructure.engines._scoring import (
    _predict_with_tree_ensemble,
    _score,
    _score_validation,
)

_MANIFEST = EngineManifest(
    id="descriptors-xgboost",
    version="1.0.0",
    name="RDKit descriptors + XGBoost",
    description=(
        "Gradient-boosted trees over the full set of RDKit physicochemical descriptors "
        "-- size, lipophilicity, polarity, topology and charge -- rather than a hashed "
        "structural fingerprint. This is the configuration behind most published ADMET "
        "leaderboard results, and it is usually the strongest option on datasets of a "
        "few hundred to a few thousand compounds."
    ),
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


class DescriptorsXGBoost:
    @staticmethod
    def manifest() -> EngineManifest:
        return _MANIFEST

    def train(self, ctx: TrainContext) -> TrainResult:
        conditions = validate_conditions(_MANIFEST, ctx.conditions)
        train_rows = ctx.frame.filter(pl.col("split") == "train")
        test_rows = ctx.frame.filter(pl.col("split") == "test")

        x_train = rdkit_descriptors(train_rows[ctx.structure_column].to_list())
        y_train = train_rows[ctx.target_column].to_numpy()
        is_classification = ctx.task is TaskType.BINARY_CLASSIFICATION

        model: XGBClassifier | XGBRegressor
        model_kwargs = {
            "n_estimators": conditions["n_estimators"],
            "max_depth": conditions["max_depth"],
            "learning_rate": conditions["learning_rate"],
            "random_state": ctx.seed,
            # Matches ecfp4_xgboost: the hist tree builder is thread-count
            # deterministic, so parallelism costs no reproducibility here.
            "n_jobs": -1,
        }
        if is_classification:
            model = XGBClassifier(**model_kwargs)
        else:
            model = XGBRegressor(**model_kwargs)
        model.fit(x_train, y_train)

        artifact = pickle.dumps(
            {
                "model": model,
                "is_classification": is_classification,
                # Both keys are what let `_predict_with_tree_ensemble` reconstruct the
                # right representation from bytes alone, and refuse if RDKit has since
                # changed which descriptors it computes.
                "featurizer": "rdkit_descriptors",
                "feature_names": DESCRIPTOR_NAMES,
            }
        )
        return TrainResult(
            artifact=artifact,
            metrics=_score(model, test_rows, ctx, is_classification, rdkit_descriptors),
            validation_metrics=_score_validation(model, ctx, is_classification, rdkit_descriptors),
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        return _predict_with_tree_ensemble(ctx)
