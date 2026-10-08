"""Protein descriptors + RandomForest. The mandatory baseline for sequence datasets.

The sequence counterpart of `ecfp4_randomforest`, and it exists for the same reason: a
Scorecard's headline number means nothing without a floor, and every engine on this
platform is measured against one. Until this existed a sequence dataset had no capable
baseline at all, so `_check_capable` refused the molecule baseline and *every* sequence
run failed -- found by training one, not by a test.

Composition and bulk physicochemistry, not embeddings: a baseline has to be a different
and simpler representation than the model it floors, or it is not a comparison. See
`protein/descriptors.py` for why it is composition-based rather than the per-position
one-hot the variant-effect literature uses, and for the licences that ruled out the
obvious descriptor packages.

No torch, so this stays on the default lane -- a sequence dataset must be trainable on a
deployment that has no GPU runner, exactly as a molecule dataset is.
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
from daikonstudio.domain.data.structure_kind import StructureKind
from daikonstudio.infrastructure.engines._options import POSITIVE_WEIGHTING, positive_weight
from daikonstudio.infrastructure.engines._scoring import (
    PROTEIN_DESCRIPTOR_FEATURIZER,
    _predict_with_tree_ensemble,
    _scored,
    bundle_features,
)
from daikonstudio.infrastructure.protein.descriptors import protein_descriptors

_MANIFEST = EngineManifest(
    id="protein-descriptors-randomforest",
    version="1.0.0",
    name="Protein descriptors + Random Forest",
    description=(
        "Amino-acid composition and bulk physicochemistry — hydropathy, charge, "
        "isoelectric point, predicted secondary-structure content — with a random "
        "forest. The required baseline every sequence model is measured against."
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
            help="More trees are steadier and slower. The default is plenty for a baseline.",
        ),
        POSITIVE_WEIGHTING,
    ),
    is_baseline=True,
    structure_kinds=(StructureKind.SEQUENCE,),
)


class ProteinDescriptorsRandomForest:
    @staticmethod
    def manifest() -> EngineManifest:
        return _MANIFEST

    def train(self, ctx: TrainContext) -> TrainResult:
        conditions = validate_conditions(_MANIFEST, ctx.conditions)
        train_rows = ctx.frame.filter(pl.col("split") == "train")
        x_train = protein_descriptors(train_rows[ctx.structure_column].to_list())
        y_train = train_rows[ctx.target_column].to_numpy()

        is_classification = ctx.task is TaskType.BINARY_CLASSIFICATION
        weight = (
            positive_weight(y_train, str(conditions["positive_weighting"]))
            if is_classification
            else None
        )

        model: RandomForestClassifier | RandomForestRegressor
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

        # Single-threaded predict, for the same reason and with the same ceiling as
        # `ecfp4_randomforest` -- read the comment there before changing this. The short
        # version: RandomForest reduces per-tree predictions in thread-completion order,
        # which drifts at ~1e-16 between runs and breaks exact reproducibility.
        model.n_jobs = 1

        artifact = pickle.dumps(
            {
                "model": model,
                "is_classification": is_classification,
                **bundle_features(PROTEIN_DESCRIPTOR_FEATURIZER),
            }
        )
        metrics, validation_metrics, cutoffs = _scored(
            model, ctx, is_classification, protein_descriptors
        )
        return TrainResult(
            artifact=artifact,
            metrics=metrics,
            validation_metrics=validation_metrics,
            cutoffs=cutoffs,
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        return _predict_with_tree_ensemble(ctx)
