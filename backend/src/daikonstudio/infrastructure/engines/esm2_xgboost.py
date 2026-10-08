"""ESM-2 650M embeddings + XGBoost -- the first engine that reads a protein sequence.

Every other engine in the roster reads a molecule: a graph (chemprop), a fingerprint
(the ECFP4 engines and the GP), a descriptor vector, or SMILES tokens (MoLFormer). This
one reads the structure column as an amino-acid sequence instead, which makes it a new
*modality* rather than a new architecture. Nothing in the Engine contract had to change
for it: `TrainContext.structure_column` is named for a structure, not for SMILES.

Frozen embeddings into a tree head, and both halves of that are deliberate.

Frozen, because the evidence against fine-tuning as a default is strong -- see
`protein/embed.py` for the FLIP2 numbers. A tree head rather than the Gaussian process
that tops supervised ProteinGym, because this is the first cut: XGBoost on a dense
feature matrix reuses `_scored` and `_predict_with_tree_ensemble` exactly as
`descriptors_xgboost.py` does, so the whole path -- featurize, fit, score, persist,
reload, predict -- is proven with almost no new code. The GP head is the follow-up, and
it is the one the literature actually favours: Kermut is a GP over these same
embeddings. Expect it to beat this engine.

What the honest claim for this engine is, and is not. Kermut's headline 0.662 mean
Spearman on supervised ProteinGym needs ProteinMPNN structural features too; its own
ablation puts the sequence-only variant at 0.594, which sits *below* ProteinNPT's
0.613-0.619. So frozen embeddings are not a leaderboard win. They are roughly ten
minutes against four hundred hours, on heads this project already ships, far above
one-hot (0.224) and ESM-2's zero-shot likelihood (0.414). Cost, not accuracy.

`lane="gpu"` because it imports torch, like chemprop and MoLFormer. It resolves to CUDA,
MPS or CPU on whatever runs it (`protein/embed.py:_loaded`); the lane is a statement
about the dependency, not about the hardware.
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
from daikonstudio.domain.data.structure_kind import StructureKind
from daikonstudio.infrastructure.engines._options import POSITIVE_WEIGHTING, positive_weight
from daikonstudio.infrastructure.engines._scoring import (
    ESM2_FEATURIZER,
    _predict_with_tree_ensemble,
    _scored,
    bundle_features,
)
from daikonstudio.infrastructure.engines.ecfp4_xgboost import fit_on_device
from daikonstudio.infrastructure.protein.embed import esm2_650m

_MANIFEST = EngineManifest(
    id="esm2-xgboost",
    version="1.0.0",
    name="ESM-2 embeddings + XGBoost",
    description=(
        "Gradient-boosted trees on frozen ESM-2 650M protein embeddings. Reads the "
        "structure column as an amino-acid sequence rather than as a molecule, so it "
        "suits proteins, enzymes and canonical peptides. The protein model is used as "
        "a fixed featurizer and is not retrained, which keeps fits to minutes."
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
            "better; deeper trees can capture more complex sequence-activity patterns.",
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
    structure_kinds=(StructureKind.SEQUENCE,),
)


class Esm2XGBoost:
    @staticmethod
    def manifest() -> EngineManifest:
        return _MANIFEST

    def train(self, ctx: TrainContext) -> TrainResult:
        conditions = validate_conditions(_MANIFEST, ctx.conditions)
        train_rows = ctx.frame.filter(pl.col("split") == "train")

        # Before the fit, not inside it: embedding is the slow half, and a run that is
        # already cancelled should not spend minutes on a forward pass first.
        ctx.report(0.0, "Embedding sequences with ESM-2")
        x_train = esm2_650m(train_rows[ctx.structure_column].to_list())
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
                **bundle_features(ESM2_FEATURIZER),
            }
        )
        metrics, validation_metrics, cutoffs = _scored(model, ctx, is_classification, esm2_650m)
        return TrainResult(
            artifact=artifact,
            metrics=metrics,
            validation_metrics=validation_metrics,
            cutoffs=cutoffs,
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        return _predict_with_tree_ensemble(ctx)
