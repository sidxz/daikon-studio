import math
import pickle
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import matthews_corrcoef

from daikonstudio.application.engines.context import (
    MIN_CUTOFF_CLASS_COUNT,
    PredictContext,
    TrainContext,
)
from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.infrastructure.chem.featurize import DESCRIPTOR_NAMES
from daikonstudio.infrastructure.engines._scoring import _scored
from daikonstudio.infrastructure.engines.ecfp4_randomforest import Ecfp4RandomForest
from daikonstudio.infrastructure.engines.ecfp4_xgboost import Ecfp4XGBoost
from tests.helpers.frames import imbalanced_frame

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


@pytest.mark.parametrize("engine", [Ecfp4XGBoost(), Ecfp4RandomForest()])
def test_train_returns_artifact_and_metrics(engine):
    ctx = TrainContext(
        frame=frame(),
        targets={"y": TaskType.REGRESSION},
        structure_column="smiles",
        conditions={},
        seed=42,
    )
    result = engine.train(ctx)
    assert isinstance(result.artifact, bytes) and len(result.artifact) > 0
    assert "rmse" in result.metrics["y"]


@pytest.mark.parametrize("engine", [Ecfp4XGBoost(), Ecfp4RandomForest()])
def test_predict_returns_one_row_per_input(engine):
    ctx = TrainContext(
        frame=frame(),
        targets={"y": TaskType.REGRESSION},
        structure_column="smiles",
        conditions={},
        seed=42,
    )
    artifact = engine.train(ctx).artifact
    predictions = engine.predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": ["CCO", "c1ccccc1"]}),
            structure_column="smiles",
            artifact=artifact,
            conditions={},
            target_columns=("y",),
        )
    )
    assert predictions.height == 2
    assert set(predictions.columns) == {"row_id", "value", "uncertainty"}


def test_training_is_reproducible_from_the_seed():
    ctx = TrainContext(
        frame=frame(),
        targets={"y": TaskType.REGRESSION},
        structure_column="smiles",
        conditions={},
        seed=42,
    )
    first = Ecfp4RandomForest().train(ctx).metrics["y"]["rmse"]
    second = Ecfp4RandomForest().train(ctx).metrics["y"]["rmse"]
    assert first == second


def test_a_regression_target_of_only_zeros_and_ones_still_trains_a_regressor():
    """Guards the sniffing bug: task comes from TargetSpec, never from the values."""
    binary_looking = pl.DataFrame(
        {
            "smiles": SMILES,
            "y": [0.0, 1.0] * 6,
            "split": SPLIT,
        }
    )
    ctx = TrainContext(
        frame=binary_looking,
        targets={"y": TaskType.REGRESSION},
        structure_column="smiles",
        conditions={},
        seed=42,
    )
    assert "rmse" in Ecfp4RandomForest().train(ctx).metrics["y"]


def test_random_forest_is_flagged_as_the_baseline():
    assert Ecfp4RandomForest.manifest().is_baseline is True
    assert Ecfp4XGBoost.manifest().is_baseline is False


@pytest.mark.parametrize("engine", [Ecfp4XGBoost(), Ecfp4RandomForest()])
def test_single_class_train_split_reports_undefined_metrics_not_a_crash(engine):
    """A single-class TRAIN split makes predict_proba return one column, not two --
    guards the IndexError this used to raise in both scoring and predict(), and
    checks that every classification metric comes back uniformly undefined rather
    than a mix of NaN and a misleadingly confident number."""
    single_class_train = pl.DataFrame(
        {
            "smiles": SMILES[:10],
            "y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "split": ["train"] * 8 + ["test"] * 2,
        }
    )
    ctx = TrainContext(
        frame=single_class_train,
        targets={"y": TaskType.BINARY_CLASSIFICATION},
        structure_column="smiles",
        conditions={},
        seed=42,
    )
    result = engine.train(ctx)
    assert set(result.metrics["y"]) == {"mcc", "balanced_accuracy", "auroc", "auprc"}
    assert all(math.isnan(value) for value in result.metrics["y"].values())
    assert "accuracy" not in result.metrics["y"]

    predictions = engine.predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": SMILES[:3]}),
            structure_column="smiles",
            artifact=result.artifact,
            conditions={},
            target_columns=("y",),
        )
    )
    assert predictions.height == 3


# --- positive weighting, descriptors and tuned cutoffs ------------------------------


def _binary_ctx(frame, **conditions):
    return TrainContext(
        frame=frame,
        targets={"y": TaskType.BINARY_CLASSIFICATION},
        structure_column="smiles",
        conditions=conditions,
        seed=1,
    )


@pytest.mark.parametrize("engine", [Ecfp4RandomForest(), Ecfp4XGBoost()])
def test_balanced_weighting_raises_the_predicted_probability_of_actives(engine):
    frame = imbalanced_frame()
    test = frame.filter(pl.col("split") == "test")

    def mean_probability(**conditions):
        artifact = engine.train(_binary_ctx(frame, **conditions)).artifact
        out = engine.predict(
            PredictContext(
                frame=test,
                structure_column="smiles",
                artifact=artifact,
                conditions={},
                target_columns=("y",),
            )
        )
        return out["value"].mean()

    assert mean_probability(positive_weighting="balanced") > mean_probability()


@pytest.mark.parametrize("engine", [Ecfp4RandomForest(), Ecfp4XGBoost()])
def test_descriptors_widen_the_input_and_round_trip(engine):
    frame = imbalanced_frame()
    result = engine.train(_binary_ctx(frame, rdkit_descriptors=True))
    bundle = pickle.loads(result.artifact)
    assert bundle["featurizer"] == "ecfp4+rdkit_descriptors"
    assert bundle["model"].n_features_in_ == 2048 + len(DESCRIPTOR_NAMES)
    out = engine.predict(
        PredictContext(
            frame=frame.head(5),
            structure_column="smiles",
            artifact=result.artifact,
            conditions={},
            target_columns=("y",),
        )
    )
    assert out.height == 5


@pytest.mark.parametrize("engine", [Ecfp4RandomForest(), Ecfp4XGBoost()])
def test_a_tuned_cutoff_is_returned_and_mcc_is_reported_at_it(engine):
    frame = imbalanced_frame()
    result = engine.train(replace(_binary_ctx(frame), tune_cutoffs=True))
    assert result.cutoffs is not None and set(result.cutoffs) == {"y"}
    cut = result.cutoffs["y"]
    test = frame.filter(pl.col("split") == "test")
    probabilities = engine.predict(
        PredictContext(
            frame=test,
            structure_column="smiles",
            artifact=result.artifact,
            conditions={},
            target_columns=("y",),
        )
    )["value"].to_numpy()
    expected = matthews_corrcoef(test["y"].to_numpy(), (probabilities >= cut).astype(int))
    assert result.metrics["y"]["mcc"] == pytest.approx(expected)


@pytest.mark.parametrize("engine", [Ecfp4RandomForest(), Ecfp4XGBoost()])
def test_without_tuning_there_are_no_cutoffs(engine):
    assert engine.train(_binary_ctx(imbalanced_frame())).cutoffs is None


class _Spy:
    """A model whose one feature is its own probability and whose `predict` is the 0.5 rule."""

    classes_ = np.array([0, 1])

    def __init__(self):
        self.predicted = False

    def predict(self, x):
        self.predicted = True
        return (x[:, 0] >= 0.5).astype(int)

    def predict_proba(self, x):
        return np.column_stack([1 - x[:, 0], x[:, 0]])


def _probability_features(smiles_list):
    return np.array([[float(text)] for text in smiles_list])


def _spy_context(validation_positives=15, **overrides):
    """Validation actives score 0.30-0.44, test actives 0.35-0.49, inactives 0.05-0.19 in both.

    Each partition is perfectly separable and all-negative at 0.5, but the MCC-optimal
    cutoff is 0.30 on validation and 0.35 on test -- so a cutoff tuned on the wrong
    partition shows. Test is still perfectly separated at the validation cutoff."""
    rows = {"smiles": [], "y": [], "split": []}
    for split, positives, lowest_active in (
        ("validation", validation_positives, 0.30),
        ("test", 15, 0.35),
    ):
        scores = [(lowest_active + 0.01 * k, 1) for k in range(positives)]
        scores += [(0.05 + 0.01 * k, 0) for k in range(15)]
        for score, label in scores:
            rows["smiles"].append(repr(score))
            rows["y"].append(label)
            rows["split"].append(split)
    return TrainContext(
        frame=pl.DataFrame(rows),
        targets={"y": TaskType.BINARY_CLASSIFICATION},
        structure_column="smiles",
        conditions={},
        seed=1,
        **overrides,
    )


def test_a_tuned_cutoff_replaces_the_hard_labels_in_test_and_validation_metrics():
    spy = _Spy()
    metrics, validation, cutoffs = _scored(
        spy, _spy_context(tune_cutoffs=True), True, _probability_features
    )
    # the validation-optimal value; tuning on the test rows would give 0.35
    assert cutoffs == {"y": pytest.approx(0.30)}
    assert metrics["y"]["mcc"] == pytest.approx(1.0)
    assert validation["y"]["mcc"] == pytest.approx(1.0)
    assert not spy.predicted


def test_without_a_cutoff_the_hard_labels_are_model_predict():
    """The default path must stay `model.predict` exactly; only a cutoff replaces it."""
    spy = _Spy()
    metrics, validation, cutoffs = _scored(spy, _spy_context(), True, _probability_features)
    assert cutoffs is None and spy.predicted
    assert metrics["y"]["mcc"] == 0.0  # everything is under 0.5, so nothing is predicted active
    assert validation["y"]["mcc"] == 0.0


def test_too_few_validation_actives_leave_the_cutoff_untuned():
    spy = _Spy()
    metrics, _, cutoffs = _scored(
        spy,
        _spy_context(validation_positives=MIN_CUTOFF_CLASS_COUNT - 1, tune_cutoffs=True),
        True,
        _probability_features,
    )
    assert cutoffs is None and spy.predicted
    assert metrics["y"]["mcc"] == 0.0


def test_an_xgboost_fit_uses_a_visible_gpu_and_ends_on_the_cpu(monkeypatch):
    """CUDA is only in prod, so the switch is checked here: fit where the GPU is, then
    score and store on the CPU, so the artifact loads on a runner with no GPU."""
    import sys
    from types import SimpleNamespace

    import numpy as np
    from xgboost import XGBClassifier

    from daikonstudio.infrastructure.engines import ecfp4_xgboost

    gpu = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", gpu)
    assert ecfp4_xgboost.xgboost_device() == "cuda"
    monkeypatch.setitem(sys.modules, "torch", None)  # the CPU image: no torch at all
    assert ecfp4_xgboost.xgboost_device() == "cpu"

    monkeypatch.setattr(ecfp4_xgboost, "xgboost_device", lambda: "cuda")
    model = XGBClassifier(n_estimators=2)
    during: list[str] = []
    monkeypatch.setattr(model, "fit", lambda x, y: during.append(model.get_params()["device"]))
    ecfp4_xgboost.fit_on_device(model, np.zeros((4, 2)), np.array([0, 1, 0, 1]))
    assert during == ["cuda"]
    assert model.get_params()["device"] == "cpu"
