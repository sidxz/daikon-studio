"""Frozen MolEncoder embeddings + XGBoost -- the chemistry mirror of `esm2_xgboost.py`.

A transformer representation of a molecule without a transformer fit. `molformer-xl`
fine-tunes 46.8M parameters for minutes to hours; this reads a 15M-parameter encoder
once, frozen, and fits boosted trees on the result in seconds. That is the whole claim:
**a cheap second opinion from a different inductive bias**, not a better one. The
published evidence behind the roster says no 2025-26 SMILES transformer beats ECFP4 plus
a forest or Chemprop under a leakage-aware split, and we have measured nothing here
against the fingerprint engines yet, so the help text promises nothing about accuracy.

Two things to know before reading a Scorecard from it. The representation is the CLS
token, whose authors note it is most meaningful after fine-tuning on the task -- and
nothing fine-tunes here (`chem/molencoder.py`). And the trees see 384 dense dimensions
with no names, so a feature attribution from this engine cannot be read the way the
descriptor engine's can.

`lane="gpu"` because it imports torch, as chemprop, MoLFormer and the ESM-2 engines do.
It resolves to CUDA, MPS or CPU on whatever runs it; the lane is a statement about the
dependency, not about the hardware.
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
from daikonstudio.infrastructure.chem.molencoder import embed as molencoder_embed
from daikonstudio.infrastructure.engines._options import POSITIVE_WEIGHTING, positive_weight
from daikonstudio.infrastructure.engines._scoring import (
    MOLENCODER_FEATURIZER,
    _predict_with_tree_ensemble,
    _scored,
    bundle_features,
)
from daikonstudio.infrastructure.engines.ecfp4_xgboost import fit_on_device

_MANIFEST = EngineManifest(
    id="molencoder-xgboost",
    version="1.0.0",
    name="MolEncoder + XGBoost",
    description=(
        "Gradient-boosted trees on frozen MolEncoder embeddings. MolEncoder is a "
        "15-million-parameter transformer pretrained on about a million ChEMBL molecules, "
        "read here as a fixed featurizer rather than retrained, so a fit takes seconds "
        "to minutes. It reads the SMILES string rather than the molecular graph or a "
        "fingerprint, which makes it a different view of a molecule; we have not measured "
        "an accuracy advantage over the fingerprint or descriptor engines."
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
    lane="gpu",
    is_baseline=False,
)


class MolEncoderXGBoost:
    @staticmethod
    def manifest() -> EngineManifest:
        return _MANIFEST

    def train(self, ctx: TrainContext) -> TrainResult:
        conditions = validate_conditions(_MANIFEST, ctx.conditions)
        train_rows = ctx.frame.filter(pl.col("split") == "train")

        # Before the fit, not inside it: embedding is the slow half, and a run that is
        # already cancelled should not spend a forward pass first.
        ctx.report(0.0, "Embedding molecules with MolEncoder")
        x_train = molencoder_embed(train_rows[ctx.structure_column].to_list())
        y_train = train_rows[ctx.target_column].to_numpy()
        ctx.report(0.5, "Fitting")

        is_classification = ctx.task is TaskType.BINARY_CLASSIFICATION
        # Regression fits ignore the setting, as in `descriptors_xgboost`: in a mixed
        # dataset fanned out per target, only the active/inactive targets are weighted.
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
            # Threads and device are `fit_on_device`'s, as in ecfp4_xgboost.
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
                # Names the representation so `_predict_with_tree_ensemble` can rebuild
                # it from bytes alone -- the predict call site knows only the artifact.
                **bundle_features(MOLENCODER_FEATURIZER),
            }
        )
        metrics, validation_metrics, cutoffs = _scored(
            model, ctx, is_classification, molencoder_embed
        )
        return TrainResult(
            artifact=artifact,
            metrics=metrics,
            validation_metrics=validation_metrics,
            cutoffs=cutoffs,
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        return _predict_with_tree_ensemble(ctx)
