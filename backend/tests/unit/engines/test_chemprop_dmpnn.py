"""The chemprop engine's contract, exercised on CPU.

Skipped unless chemprop imports, because the default-lane worker and the API tier
deliberately install without it. Two epochs on twenty molecules: this is not a test of
whether a D-MPNN learns anything, it is a test that the adapter honours the Engine
contract -- metric names the Scorecard can compare, and a predict frame whose dtypes
line up with every other engine's.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("chemprop")

from daikonstudio.application.engines.context import PredictContext, TrainContext
from daikonstudio.application.engines.manifest import ConditionType, TaskType
from daikonstudio.infrastructure.engines.chemprop_dmpnn import ChempropDMPNN

_SMILES = [
    "CCO",
    "CCC",
    "CCCC",
    "c1ccccc1",
    "CC(=O)O",
    "CCN",
    "CCOCC",
    "CC(C)O",
    "c1ccncc1",
    "CCCCO",
    "CC(C)(C)O",
    "CCCCCC",
    "c1ccc(cc1)O",
    "CCS",
    "CCCl",
    "CC=O",
    "CCC(=O)O",
    "c1ccc(cc1)N",
    "CCCCCCC",
    "CC(C)C",
]
_SPLITS = ["train"] * 12 + ["validation"] * 4 + ["test"] * 4
_FAST = {"epochs": 2, "depth": 2, "message_hidden_dim": 64, "batch_size": 8}


def _frame(targets: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"smiles": _SMILES, "y": targets, "split": _SPLITS})


def _train_context(frame: pl.DataFrame, task: TaskType) -> TrainContext:
    return TrainContext(
        frame=frame,
        task=task,
        structure_column="smiles",
        target_column="y",
        conditions=_FAST,
        seed=13,
    )


def test_manifest_declares_the_gpu_lane() -> None:
    """The whole point of this engine: it must not land on the default queue, where a
    worker without CUDA would pick it up."""
    assert ChempropDMPNN.manifest().lane == "gpu"


def test_regression_reports_the_shared_metric_vocabulary() -> None:
    frame = _frame([float(i) for i in range(20)])

    result = ChempropDMPNN().train(_train_context(frame, TaskType.REGRESSION))

    assert sorted(result.metrics) == ["mae", "r2", "rmse"]
    assert result.artifact  # a loadable checkpoint, not an empty blob


def test_classification_reports_the_shared_metric_vocabulary() -> None:
    frame = _frame([float(i % 2) for i in range(20)])

    result = ChempropDMPNN().train(_train_context(frame, TaskType.BINARY_CLASSIFICATION))

    assert sorted(result.metrics) == ["auprc", "auroc", "balanced_accuracy", "mcc"]


def test_predict_returns_the_contracted_schema_and_dtypes() -> None:
    """An all-None uncertainty column inferred as polars Null would make this engine's
    output schema-incompatible with every other engine's."""
    frame = _frame([float(i) for i in range(20)])
    trained = ChempropDMPNN().train(_train_context(frame, TaskType.REGRESSION))

    predictions = ChempropDMPNN().predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": ["CCO", "CCC"]}),
            structure_column="smiles",
            artifact=trained.artifact,
            conditions={},
        )
    )

    assert predictions.columns == ["row_id", "value", "uncertainty"]
    assert predictions.schema["row_id"] == pl.Int64
    assert predictions.schema["value"] == pl.Float64
    assert predictions.schema["uncertainty"] == pl.Float64
    assert predictions.height == 2


def test_classification_predictions_are_probabilities() -> None:
    """chemprop's BinaryClassificationFFN already applies sigmoid in forward(). A
    second one here would squash every prediction into [0.5, 0.73]."""
    frame = _frame([float(i % 2) for i in range(20)])
    trained = ChempropDMPNN().train(_train_context(frame, TaskType.BINARY_CLASSIFICATION))

    predictions = ChempropDMPNN().predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": _SMILES}),
            structure_column="smiles",
            artifact=trained.artifact,
            conditions={},
        )
    )

    values = predictions["value"].to_list()
    assert all(0.0 <= value <= 1.0 for value in values)


def test_report_is_called_once_per_epoch() -> None:
    """This callback is the only thing that can stop a GPU fit already in flight."""
    calls: list[tuple[float, str]] = []
    frame = _frame([float(i) for i in range(20)])
    ctx = TrainContext(
        frame=frame,
        task=TaskType.REGRESSION,
        structure_column="smiles",
        target_column="y",
        conditions=_FAST,
        seed=13,
        report=lambda fraction, phase: calls.append((fraction, phase)),
    )

    ChempropDMPNN().train(ctx)

    assert len(calls) == 2
    assert calls[-1][0] == pytest.approx(1.0)
    assert calls[-1][1] == "training chemprop-dmpnn"


def test_the_manifest_offers_pretrained_weights():
    spec = next(c for c in ChempropDMPNN.manifest().conditions if c.key == "pretrained")
    assert spec.type is ConditionType.ENUM
    assert spec.default == "none"
    assert spec.options == ("none", "CheMeleon")


@pytest.mark.skipif(
    not (
        Path(
            os.environ.get("STUDIO_PRETRAINED_WEIGHTS_DIR", "~/.cache/daikon-studio/weights")
        ).expanduser()
        / "chemeleon_mp.pt"
    ).exists(),
    reason="CheMeleon weights not cached; CI does not download 35 MB",
)
def test_chemeleon_builds_a_network_sized_by_the_checkpoint_not_the_conditions():
    """CheMeleon pins d_h=2048. The FFN's input_dim must follow the checkpoint,
    not the message_hidden_dim condition, or the first layer is built for the
    wrong width and the fit dies on a shape mismatch."""
    from daikonstudio.infrastructure.engines.chemprop_dmpnn import _build_model

    model = _build_model(
        pretrained="CheMeleon",
        weights_dir=os.environ.get(
            "STUDIO_PRETRAINED_WEIGHTS_DIR", "~/.cache/daikon-studio/weights"
        ),
        hidden=300,
        depth=3,
        is_classification=False,
        output_transform=None,
    )
    assert model.message_passing.output_dim == 2048
