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

Runs on the default lane: RDKit plus XGBoost, no torch. A runner that has a GPU fits on it
(`ecfp4_xgboost.fit_on_device`).
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
from daikonstudio.infrastructure.engines._options import POSITIVE_WEIGHTING, positive_weight
from daikonstudio.infrastructure.engines._scoring import _predict_with_tree_ensemble, _scored
from daikonstudio.infrastructure.engines.ecfp4_xgboost import fit_on_device

_MANIFEST = EngineManifest(
    id="descriptors-xgboost",
    version="1.0.0",
    name="RDKit descriptors + XGBoost",
    description=(
        "Gradient-boosted trees on RDKit 2D descriptors (size, lipophilicity, polarity, "
        "topology and charge) instead of a hashed fingerprint. A strong, widely used "
        "configuration for ADMET endpoints, particularly on datasets of a few hundred to "
        "a few thousand compounds."
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
        POSITIVE_WEIGHTING,
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

        x_train = rdkit_descriptors(train_rows[ctx.structure_column].to_list())
        y_train = train_rows[ctx.target_column].to_numpy()
        is_classification = ctx.task is TaskType.BINARY_CLASSIFICATION
        # Regression fits ignore the setting: in a mixed dataset fanned out per target,
        # only the active/inactive targets are weighted.
        weight = (
            positive_weight(y_train, str(conditions["positive_weighting"]))
            if is_classification
            else None
        )

        model: XGBClassifier | XGBRegressor
        model_kwargs = {
            "n_estimators": conditions["n_estimators"],
            "max_depth": conditions["max_depth"],
            "learning_rate": conditions["learning_rate"],
            "random_state": ctx.seed,
            # Matches ecfp4_xgboost: the hist tree builder is thread-count
            # deterministic, so parallelism costs no reproducibility here. On a GPU
            # runner the fit runs on CUDA instead (`fit_on_device`).
            "n_jobs": -1,
        }
        if is_classification:
            if weight is not None:
                model_kwargs["scale_pos_weight"] = weight
            model = XGBClassifier(**model_kwargs)
        else:
            model = XGBRegressor(**model_kwargs)
        fit_on_device(model, x_train, y_train)

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
        metrics, validation_metrics, cutoffs = _scored(
            model, ctx, is_classification, rdkit_descriptors
        )
        return TrainResult(
            artifact=artifact,
            metrics=metrics,
            validation_metrics=validation_metrics,
            cutoffs=cutoffs,
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        return _predict_with_tree_ensemble(ctx)
