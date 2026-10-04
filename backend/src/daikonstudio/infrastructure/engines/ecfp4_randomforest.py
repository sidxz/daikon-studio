"""ECFP4 + RandomForest. The mandatory baseline every Scorecard compares against.

In the Polaris ADMET competition, fingerprint-plus-random-forest baselines placed
around 20th of 66 teams. A scientist whose model cannot beat this needs to know on
the first screen, so this has to be a genuinely competitive model, not a strawman.
"""

from __future__ import annotations

import pickle

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
            help="Number of decision trees in the ensemble. More trees give more stable "
            "predictions but take longer to train.",
        ),
        POSITIVE_WEIGHTING,
        RDKIT_DESCRIPTORS,
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

        # No annotation on `model`: sklearn ships no py.typed marker, so both
        # constructors already resolve to Any (see the import above) -- declaring a
        # union of the two here would just be a union of Any, which mypy rejects as
        # a type. Any is what falls out naturally by leaving it unannotated.
        #
        # n_jobs=-1 for fitting: per-tree random states are drawn upfront, so which
        # thread builds which tree doesn't affect the result -- fitting in parallel
        # is bit-identical to fitting serially, and is the expensive half (measured
        # 5.6-8x slower single-threaded on realistic assay sizes).
        if is_classification:
            model = RandomForestClassifier(
                n_estimators=conditions["n_estimators"],
                random_state=ctx.seed,
                n_jobs=-1,
                class_weight=None if weight is None else {0: 1.0, 1: weight},
            )
        else:
            model = RandomForestRegressor(
                n_estimators=conditions["n_estimators"], random_state=ctx.seed, n_jobs=-1
            )
        model.fit(x_train, y_train)

        # ponytail: predict() pinned to single-threaded, forever, on this fitted
        # model. Unlike fitting, RandomForest's predict() sums per-tree outputs
        # under a lock in thread-completion order, which is not fixed run to run --
        # measured a ~3e-16 relative drift between two n_jobs=-1 predict calls on
        # identical input, enough to break exact-equality reproducibility. Must be
        # set before scoring below AND before the artifact is pickled, so the
        # persisted model also predicts reproducibly once loaded back in production.
        # Ceiling: predicting a very large batch runs on one core. Upgrade path if
        # that ever matters: parallelize per-tree predictions manually (e.g.
        # joblib.Parallel over model.estimators_) and reduce them in a fixed
        # tree-index order, instead of relying on RandomForest's internal reduction.
        model.n_jobs = 1

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
