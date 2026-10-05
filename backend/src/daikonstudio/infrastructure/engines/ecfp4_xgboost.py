"""ECFP4 + XGBoost. The gradient-boosted alternative to the RandomForest baseline."""

from __future__ import annotations

import pickle
from typing import Any

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
from daikonstudio.infrastructure.engines._options import (
    POSITIVE_WEIGHTING,
    RDKIT_DESCRIPTORS,
    positive_weight,
)
from daikonstudio.infrastructure.engines._scoring import (
    _predict_with_tree_ensemble,
    _scored,
    bundle_features,
    tree_featurizer,
)

_MANIFEST = EngineManifest(
    id="ecfp4-xgboost",
    version="1.0.0",
    name="ECFP4 + XGBoost",
    description="Morgan fingerprints with gradient-boosted trees. Often more accurate "
    "than the random-forest baseline, but more sensitive to hyperparameters.",
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
        RDKIT_DESCRIPTORS,
    ),
    is_baseline=False,
)


def xgboost_device() -> str:
    """Where an XGBoost fit runs: "cuda" when this process can see a GPU, else "cpu".

    A tree baseline on the GPU runner ran on one CPU core -- that image pins
    OMP_NUM_THREADS=1 (Dockerfile.gpu), so n_jobs=-1 meant one thread -- for as long as
    the neural fit before it, with the GPU idle (prod, 2026-10-05). The pinned xgboost
    wheel is the CUDA build. On atlantic (10k nuisance set, 400 trees) a fit took 0.8 s
    on CUDA against 4.2 s on that one core, with the same test AUROC and PR AUC to three
    decimals and 0.08 GB of GPU memory. Asked through torch, which only the GPU image
    has; the CPU image cannot import it, and has no GPU to find either.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def fit_on_device(model: XGBClassifier | XGBRegressor, x: Any, y: Any) -> None:
    """Fit where `xgboost_device` says, then move the model to the CPU for everything
    after: scoring, and the artifact, which must load on a runner with no GPU. Scoring
    numpy arrays with a CUDA model would only fall back to the CPU with a warning. CUDA
    fits repeat bit-identically, measured; they can differ from a CPU fit by up to 0.01
    in a predicted probability, so where a model trains is part of what it is."""
    model.set_params(device=xgboost_device())
    model.fit(x, y)
    model.set_params(device="cpu")


class Ecfp4XGBoost:
    @staticmethod
    def manifest() -> EngineManifest:
        return _MANIFEST

    def train(self, ctx: TrainContext) -> TrainResult:
        conditions = validate_conditions(_MANIFEST, ctx.conditions)
        train_rows = ctx.frame.filter(pl.col("split") == "train")

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
            # On a GPU runner the fit runs on CUDA instead (`fit_on_device`).
            "n_jobs": -1,
        }
        if is_classification:
            if weight is not None:
                model_kwargs["scale_pos_weight"] = weight
            model = XGBClassifier(**model_kwargs)
        else:
            model = XGBRegressor(**model_kwargs)
        fit_on_device(model, x_train, y_train)

        # pickle.dumps serializes the fitted model; safe to write, since only our
        # own predict() ever reads this artifact back (see _scoring.py for the load
        # side, and its comment on why deserializing it is safe there).
        artifact = pickle.dumps(
            {
                "model": model,
                "is_classification": is_classification,
                **bundle_features(featurizer_key),
            }
        )
        metrics, validation_metrics, cutoffs = _scored(model, ctx, is_classification, featurizer)
        return TrainResult(
            artifact=artifact,
            metrics=metrics,
            validation_metrics=validation_metrics,
            cutoffs=cutoffs,
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        return _predict_with_tree_ensemble(ctx)
