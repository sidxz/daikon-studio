"""ECFP4 + LightGBM. The roster's second gradient-boosted tree, on sparse fingerprints.

Why a second boosting library rather than another condition on `ecfp4-xgboost`: the
two grow trees differently. XGBoost grows level-wise to a fixed depth; LightGBM grows
leaf-wise, splitting whichever leaf reduces the loss most, and bundles mutually
exclusive sparse features (its EFB) before it starts. A 2048-bit Morgan fingerprint is
almost entirely zeros, which is the case EFB was designed for -- so this pairing is
where the two libraries diverge most, and it is the one worth having. On dense RDKit
descriptors they converge to near-identical models, which is why there is no
`descriptors-lightgbm` to go with `descriptors-xgboost`.

Everything else -- featurization, scoring, the artifact bundle and the predict path --
is shared with the other tree engines through `_scoring.py`. LightGBM's sklearn wrapper
exposes no `estimators_`, so `_predict_with_tree_ensemble` finds no ensemble spread and
reports `uncertainty` as null, exactly as it does for XGBoost. That is correct rather
than lazy: boosted trees are additive corrections to one running prediction, not
independent estimates of it, so their spread is not an uncertainty.
"""

from __future__ import annotations

import pickle
from typing import Any

import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor

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
    tree_threads,
)
from daikonstudio.infrastructure.engines._scoring import (
    _predict_with_tree_ensemble,
    _scored,
    bundle_features,
    tree_featurizer,
)

_MANIFEST = EngineManifest(
    id="ecfp4-lightgbm",
    version="1.0.0",
    name="ECFP4 + LightGBM",
    description="Morgan fingerprints with leaf-wise gradient boosting. Trees grow toward "
    "the split with the largest gain rather than to a fixed depth, and sparse "
    "fingerprint input is handled natively. Usually the fastest engine on large "
    "datasets.",
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
            key="num_leaves",
            label="Maximum leaves per tree",
            type=ConditionType.INTEGER,
            default=31,
            minimum=2,
            maximum=1024,
            help="How complex each tree may become. This is LightGBM's main capacity "
            "control in place of a depth limit, because it grows leaf-wise. Larger "
            "values fit more detail and overfit sooner on small datasets.",
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
        ConditionSpec(
            key="min_child_samples",
            label="Minimum compounds per leaf",
            type=ConditionType.INTEGER,
            default=20,
            minimum=1,
            maximum=200,
            help="A leaf must be supported by at least this many compounds. Raising it "
            "is the most direct guard against fitting single molecules; lower it on "
            "small assays, where the default can stop the trees splitting at all.",
        ),
        POSITIVE_WEIGHTING,
        RDKIT_DESCRIPTORS,
    ),
    is_baseline=False,
)


class Ecfp4LightGBM:
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

        model: LGBMClassifier | LGBMRegressor
        # `Any` because LightGBM's constructor is precisely typed and this dict is
        # heterogeneous -- the same erasure `_scoring.py` documents for the sklearn side.
        model_kwargs: dict[str, Any] = {
            "n_estimators": conditions["n_estimators"],
            "num_leaves": conditions["num_leaves"],
            "learning_rate": conditions["learning_rate"],
            "min_child_samples": conditions["min_child_samples"],
            "random_state": ctx.seed,
            # Never n_jobs=-1, which LightGBM turns into one thread per LOGICAL CPU (it
            # sets OpenMP's limit itself, past OMP_NUM_THREADS): its threads meet at a
            # barrier many times per tree, and on atlantic (48 CPUs, two busy elsewhere)
            # that stalled a 1-second fit to 220 s with 46 cores pegged. `tree_threads`
            # also keeps it at one thread once torch is loaded: LightGBM at 8 threads
            # after a torch op segfaults on macOS. The model predicts on one thread.
            "n_jobs": tree_threads(),
            # Unlike XGBoost's hist builder (see ecfp4_xgboost.py, which measured
            # itself bit-identical at n_jobs=-1), LightGBM's multithreaded histogram
            # construction sums floats in thread-completion order and is NOT
            # reproducible by default -- the same seed and data give slightly
            # different trees per run. `deterministic` fixes that, and LightGBM
            # requires one of the force_*_wise flags alongside it or it warns and
            # ignores the request. Row-wise is the right one here: 2048 features
            # against a training set that is usually much larger.
            #
            # The cost is real but small, and it is the same trade the RandomForest
            # engine already makes by pinning predict to n_jobs=1 -- a Scorecard that
            # changes when nothing changed is worse than a fit that takes longer.
            "deterministic": True,
            "force_row_wise": True,
            # LightGBM logs "No further splits with positive gain" per boosting round
            # once a small or saturated dataset stops splitting, which on a worker
            # means thousands of lines that say nothing actionable. The condition it
            # reports is visible in the metrics.
            "verbose": -1,
        }
        if is_classification:
            if weight is not None:
                model_kwargs["scale_pos_weight"] = weight
            model = LGBMClassifier(**model_kwargs)
        else:
            model = LGBMRegressor(**model_kwargs)
        model.fit(x_train, y_train)
        # Scoring below, and every later predict, in whatever process loads it: one
        # thread, so a prediction never runs LightGBM's OpenMP beside torch's.
        model.set_params(n_jobs=1)

        # Same bundle shape as the other tree engines, so `_load_bundle` reads it
        # back without knowing which engine wrote it. Safe to pickle for the reason
        # given in `_scoring.py`: only our own predict() ever loads this artifact.
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
