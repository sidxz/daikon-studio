import itertools
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import matthews_corrcoef

from daikonstudio.application.engines.context import PredictContext, TrainContext
from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.chem.featurize import ecfp4
from daikonstudio.infrastructure.engines.tanimoto_gp import (
    _MAX_TRAINING_ROWS,
    TanimotoGP,
    TanimotoKernel,
)

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
        targets={"y": task},
        structure_column="smiles",
        conditions=overrides.pop("conditions", {"n_restarts_optimizer": 0}),
        seed=42,
        **overrides,
    )


def predict(artifact: bytes, smiles: list[str]) -> pl.DataFrame:
    return TanimotoGP().predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": smiles}),
            structure_column="smiles",
            artifact=artifact,
            conditions={},
            target_columns=("y",),
        )
    )


# --- the kernel itself -------------------------------------------------------------


def test_kernel_matches_jaccard_computed_independently():
    """The definition, checked against a boolean set operation rather than against
    itself: intersection over union of the bits actually set."""
    x = ecfp4(["CCO", "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"])
    bits = x.astype(bool)
    k = TanimotoKernel()(x)
    for i in range(len(x)):
        for j in range(len(x)):
            intersection = int(np.logical_and(bits[i], bits[j]).sum())
            union = int(np.logical_or(bits[i], bits[j]).sum())
            assert k[i, j] == pytest.approx(intersection / union)


def test_kernel_does_not_accumulate_in_the_uint8_it_is_handed():
    """`ecfp4` returns uint8 and sklearn's dtype="numeric" validation preserves it. The
    sum of two molecules' bit counts is what overflows first, so this uses rows dense
    enough that a uint8 union would wrap -- the wrap is silent, unlike the divide."""
    dense = np.zeros((2, 2048), dtype=np.uint8)
    dense[0, :200] = 1
    dense[1, 100:300] = 1  # union 300, intersection 100 -> 1/3, but 300 wraps to 44

    k = TanimotoKernel()(dense)
    assert k[0, 1] == pytest.approx(100 / 300)
    assert np.diag(k) == pytest.approx([1.0, 1.0])


def test_kernel_is_positive_semi_definite():
    """What makes this a kernel rather than a similarity function a GP was talked into
    accepting. A negative eigenvalue means the fit is solving something else."""
    rng = np.random.default_rng(0)
    x = (rng.random((120, 2048)) < 0.02).astype(np.uint8)
    k = TanimotoKernel()(x)
    assert np.allclose(k, k.T)
    assert np.linalg.eigvalsh(k).min() > -1e-8


def test_unparseable_structure_is_zero_similarity_not_nan():
    """`ecfp4` gives an all-zero row for a SMILES RDKit cannot parse, so the union is 0
    and the ratio is 0/0. A NaN there propagates through the whole kernel matrix."""
    k = TanimotoKernel()(ecfp4(["CCO", "not-a-molecule", "not-a-molecule-either"]))
    assert not np.isnan(k).any()
    assert k[1, 1] == 0.0
    assert k[1, 2] == 0.0


def test_diag_agrees_with_the_full_matrix_it_shortcuts():
    x = ecfp4(["CCO", "not-a-molecule", "c1ccccc1"])
    assert TanimotoKernel().diag(x) == pytest.approx(np.diag(TanimotoKernel()(x)))


# --- the engine --------------------------------------------------------------------


def test_train_returns_artifact_and_metrics():
    result = TanimotoGP().train(context())
    assert isinstance(result.artifact, bytes) and len(result.artifact) > 0
    assert "rmse" in result.metrics["y"]
    assert result.validation_metrics is None  # no validation rows in this frame


def test_validation_partition_is_scored():
    rows = frame().with_columns(
        pl.Series("split", ["train"] * 6 + ["validation"] * 3 + ["test"] * 3)
    )
    result = TanimotoGP().train(context(frame=rows))
    assert result.validation_metrics is not None
    assert "rmse" in result.validation_metrics["y"]


def test_regression_uncertainty_is_a_real_posterior_spread():
    """The reason this engine exists. Not null (which is what the tree path would have
    returned for a model with no `estimators_`), non-negative, and *larger* for a
    molecule unlike anything in the training set than for one that is in it -- which a
    constant or a fabricated number would not be."""
    artifact = TanimotoGP().train(context()).artifact
    predictions = predict(artifact, ["CCO", "[Pt](Cl)(Cl)(N)N"])

    spread = predictions["uncertainty"].to_list()
    assert predictions["uncertainty"].null_count() == 0
    assert all(s >= 0.0 for s in spread)
    assert spread[1] > spread[0], spread


def test_predict_round_trips_and_tolerates_an_unparseable_structure():
    artifact = TanimotoGP().train(context()).artifact
    predictions = predict(artifact, ["CCO", "not-a-molecule", "c1ccccc1"])
    assert predictions.height == 3
    assert set(predictions.columns) == {"row_id", "value", "uncertainty"}
    assert not np.isnan(predictions["value"].to_numpy()).any()


def test_duplicate_structures_do_not_make_the_fit_singular():
    """Identical rows give a singular kernel matrix. The fitted WhiteKernel noise term
    is what keeps the Cholesky from failing, so this is that term's regression test --
    duplicate measurements of one compound are ordinary in an assay."""
    rows = pl.DataFrame(
        {
            "smiles": ["CCO"] * 4 + ["c1ccccc1"] * 4 + ["CCN", "CCCN", "CCO", "c1ccccc1"],
            "y": [1.0, 1.1, 0.9, 1.05, 5.0, 5.1, 4.9, 5.05, 2.0, 2.2, 1.0, 5.0],
            "split": ["train"] * 8 + ["test"] * 4,
        }
    )
    result = TanimotoGP().train(context(frame=rows))
    assert np.isfinite(result.metrics["y"]["rmse"])


def test_classification_trains_and_predicts():
    rows = frame().with_columns(pl.Series("y", [0, 1] * 6))
    result = TanimotoGP().train(context(TaskType.BINARY_CLASSIFICATION, frame=rows))
    assert set(result.metrics["y"]) == {"mcc", "balanced_accuracy", "auroc", "auprc"}
    assert "accuracy" not in result.metrics["y"]

    predictions = predict(result.artifact, ["CCO", "c1ccccc1"])
    values = predictions["value"].to_list()
    assert all(0.0 <= v <= 1.0 for v in values)
    assert predictions["uncertainty"].null_count() == 0


def test_single_class_training_split_is_refused_with_a_legible_message():
    """GaussianProcessClassifier raises where the tree engines fit happily and let
    `_scored` report undefined metrics. Its own message names neither the engine nor the
    fix, so the run would fail on a bare sklearn ValueError."""
    rows = frame().with_columns(pl.Series("y", [0] * 8 + [0, 1, 0, 1]))
    with pytest.raises(ValidationError, match="same label"):
        TanimotoGP().train(context(TaskType.BINARY_CLASSIFICATION, frame=rows))


def test_training_set_above_the_ceiling_is_refused_before_it_allocates():
    """O(n^2) memory and O(n^3) time: past the ceiling the failure is an OOM-killed
    worker with no message. The check runs before featurization, so this stays fast."""
    n = _MAX_TRAINING_ROWS + 1
    rows = pl.DataFrame(
        {"smiles": ["CCO"] * n, "y": [1.0] * n, "split": ["train"] * n},
    )
    with pytest.raises(ValidationError, match="accepts at most"):
        TanimotoGP().train(context(frame=rows))


def test_manifest_declares_the_cubic_ceiling_to_the_user():
    """Plan item: the ceiling is the one thing a scientist must know before choosing
    this engine, and it belongs on screen rather than in this file."""
    description = TanimotoGP.manifest().description
    assert "5,000" in description and "10,000" in description


def test_a_tuned_cutoff_is_returned_and_mcc_is_reported_at_it():
    """120 rows, because the GP is slow; validation holds 12 of each class, enough to tune.

    Anilines are active, phenols inactive, over 60 distinct substituent pairs. Not alkyl
    chains: their fingerprints are near-duplicates, which leaves the classifier's Laplace
    approximation with a negative latent variance and NaN probabilities on unseen rows.
    """
    groups = ["C", "CC", "CCC", "Cl", "Br", "F", "OC", "C#N"]
    pairs = list(itertools.product(groups, groups))[:60]
    rows = pl.DataFrame(
        {
            "smiles": [f"{head}c1cc({a})cc({b})c1" for a, b in pairs for head in "NO"],
            "y": [1, 0] * 60,
            "split": ["train", "validation", "test", "train", "train"] * 24,
        }
    )
    ctx = replace(context(TaskType.BINARY_CLASSIFICATION, frame=rows), tune_cutoffs=True)
    result = TanimotoGP().train(ctx)

    assert result.cutoffs is not None and set(result.cutoffs) == {"y"}
    test = rows.filter(pl.col("split") == "test")
    probabilities = predict(result.artifact, test["smiles"].to_list())["value"].to_numpy()
    expected = matthews_corrcoef(
        test["y"].to_numpy(), (probabilities >= result.cutoffs["y"]).astype(int)
    )
    assert result.metrics["y"]["mcc"] == pytest.approx(expected)
