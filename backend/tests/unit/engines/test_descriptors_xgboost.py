import pickle

import numpy as np
import polars as pl
import pytest

from daikonstudio.application.engines.context import PredictContext, TrainContext
from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.chem.featurize import (
    DESCRIPTOR_NAMES,
    ecfp4,
    rdkit_descriptors,
)
from daikonstudio.infrastructure.engines.descriptors_xgboost import DescriptorsXGBoost

SMILES = [
    "CCO",
    "CCCO",
    "CCCCO",
    "c1ccccc1",
    "Cc1ccccc1",
    "CCc1ccccc1",
    "CCN",
    "CCCN",
    "CCCCN",
    "c1ccncc1",
    "Cc1ccncc1",
    "CCc1ccncc1",
]
VALUES = [1.0, 1.2, 1.4, 5.0, 5.2, 5.4, 2.0, 2.2, 2.4, 6.0, 6.2, 6.4]
SPLIT = ["train"] * 8 + ["test"] * 4


def frame() -> pl.DataFrame:
    return pl.DataFrame({"smiles": SMILES, "y": VALUES, "split": SPLIT})


def context(task: TaskType = TaskType.REGRESSION, **overrides) -> TrainContext:
    return TrainContext(
        frame=overrides.pop("frame", frame()),
        task=task,
        structure_column="smiles",
        target_column="y",
        conditions={},
        seed=42,
        **overrides,
    )


def test_featurizer_is_wider_than_the_nine_profile_descriptors():
    """Guards the module mix-up: `chem/descriptors.py` is the nine a chemist reads,
    `rdkit_descriptors` is the full model input. Importing the wrong one would still
    train, just far worse."""
    x = rdkit_descriptors(SMILES)
    assert x.shape == (len(SMILES), len(DESCRIPTOR_NAMES))
    assert len(DESCRIPTOR_NAMES) > 100


def test_unparseable_smiles_yields_an_all_nan_row_not_zeros():
    """0.0 is a legitimate value for many descriptors, so a zero row would be a
    molecule that reads as measured. NaN is the only honest encoding of 'never
    computed' -- and the one XGBoost understands as missing."""
    x = rdkit_descriptors(["CCO", "not-a-molecule", "c1ccccc1"])
    assert np.isnan(x[1]).all()
    assert not np.isnan(x[0]).all()
    assert not np.isnan(x[2]).all()


def test_no_infinities_survive_the_featurizer():
    """XGBoost rejects inf outright ('Input data contains `inf`') while accepting NaN,
    and some RDKit descriptors (Ipc most notoriously) overflow on larger molecules. A
    long chain and a fused polycyclic are the shapes that trigger it."""
    x = rdkit_descriptors(["C" * 60, "c1ccc2cc3ccc4ccccc4c3cc2c1", "CCO"])
    assert not np.isinf(x).any()


def test_descriptors_differ_from_the_fingerprint_representation():
    """Cheap guard that the engine is not quietly the ECFP4 one under a new name."""
    assert rdkit_descriptors(SMILES).shape[1] != ecfp4(SMILES).shape[1]


def test_train_returns_artifact_and_metrics():
    result = DescriptorsXGBoost().train(context())
    assert isinstance(result.artifact, bytes) and len(result.artifact) > 0
    assert "rmse" in result.metrics
    assert result.validation_metrics is None  # no validation rows in this frame


def test_validation_is_scored_through_the_same_featurizer():
    """`_score_validation` defaults to ecfp4; passing the wrong featurizer there would
    not raise, it would silently score this model against 2048 fingerprint bits."""
    rows = frame().with_columns(
        pl.Series("split", ["train"] * 6 + ["validation"] * 3 + ["test"] * 3)
    )
    result = DescriptorsXGBoost().train(context(frame=rows))
    assert result.validation_metrics is not None
    assert "rmse" in result.validation_metrics


def test_predict_round_trips_and_tolerates_an_unparseable_structure():
    artifact = DescriptorsXGBoost().train(context()).artifact
    predictions = DescriptorsXGBoost().predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": ["CCO", "not-a-molecule", "c1ccccc1"]}),
            structure_column="smiles",
            artifact=artifact,
            conditions={},
        )
    )
    assert predictions.height == 3
    assert set(predictions.columns) == {"row_id", "value", "uncertainty"}
    # XGBoost has no per-tree spread to report, matching the ECFP4 XGBoost engine.
    assert predictions["uncertainty"].is_null().all()


def test_classification_trains_and_predicts():
    rows = frame().with_columns(pl.Series("y", [0, 1] * 6))
    result = DescriptorsXGBoost().train(context(TaskType.BINARY_CLASSIFICATION, frame=rows))
    assert set(result.metrics) == {"mcc", "balanced_accuracy", "auroc", "auprc"}
    assert "accuracy" not in result.metrics


def test_predicting_through_drifted_descriptors_refuses_instead_of_guessing():
    """The silent failure this exists to prevent: RDKit adds a descriptor, every stored
    model's columns shift by one, and XGBoost returns confident nonsense with no error.
    Simulated by rewriting the artifact's stored names, which is what an RDKit upgrade
    does from the model's point of view."""
    artifact = DescriptorsXGBoost().train(context()).artifact
    bundle = pickle.loads(artifact)
    bundle["feature_names"] = DESCRIPTOR_NAMES[:-1]

    with pytest.raises(ValidationError, match="different set of molecular descriptors"):
        DescriptorsXGBoost().predict(
            PredictContext(
                frame=pl.DataFrame({"smiles": ["CCO"]}),
                structure_column="smiles",
                artifact=pickle.dumps(bundle),
                conditions={},
            )
        )


def test_an_ecfp4_artifact_without_the_new_keys_still_predicts():
    """Backward compatibility is the whole reason `predict` defaults the featurizer
    rather than reading `bundle["featurizer"]`: every artifact written before this
    engine existed lacks both keys and must keep working."""
    from daikonstudio.infrastructure.engines.ecfp4_xgboost import Ecfp4XGBoost

    artifact = Ecfp4XGBoost().train(context()).artifact
    assert "featurizer" not in pickle.loads(artifact)

    predictions = Ecfp4XGBoost().predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": ["CCO", "c1ccccc1"]}),
            structure_column="smiles",
            artifact=artifact,
            conditions={},
        )
    )
    assert predictions.height == 2
