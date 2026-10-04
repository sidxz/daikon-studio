"""The MoLFormer engine's contract, on whatever device this machine has.

Skipped unless transformers imports and the checkpoint is already cached, because the
default-lane worker and the API tier deliberately install without the gpu extra, and
CI does not download 179 MB. Two epochs on twelve molecules: this is not a test of
whether a transformer learns anything, it is a test that the adapter honours the
Engine contract and that the two traps this checkpoint carries stay guarded --
reproducible inference, and predictions in the target's own units.

Nothing here pins the device. `accelerator="auto"` resolves to MPS on Apple Silicon,
CUDA on a GPU runner and CPU otherwise, and all three must satisfy the same contract.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("transformers")

from daikonstudio.application.engines.context import PredictContext, TrainContext
from daikonstudio.application.engines.manifest import ConditionType, TaskType
from daikonstudio.infrastructure.engines.molformer_xl import MolformerXL

_WEIGHTS_DIR = Path(
    os.environ.get("STUDIO_PRETRAINED_WEIGHTS_DIR", "~/.cache/daikon-studio/weights")
).expanduser()
_SNAPSHOT = _WEIGHTS_DIR / "hub" / "models--ibm-research--MoLFormer-XL-both-10pct"

needs_weights = pytest.mark.skipif(
    not _SNAPSHOT.exists(),
    reason="MoLFormer-XL weights not cached; CI does not download 179 MB",
)

_SMILES = [
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
_SPLITS = ["train"] * 6 + ["validation"] * 3 + ["test"] * 3
_FAST = {"epochs": 2, "batch_size": 8, "freeze_encoder": True}


def _frame(targets: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"smiles": _SMILES, "y": targets, "split": _SPLITS})


def _context(frame: pl.DataFrame, task: TaskType, **conditions: object) -> TrainContext:
    return TrainContext(
        frame=frame,
        targets={"y": task},
        structure_column="smiles",
        conditions={**_FAST, **conditions},
        seed=13,
    )


def test_manifest_declares_the_gpu_lane() -> None:
    """The whole point of this engine: it must not land on the default queue, where a
    worker without the gpu extra would pick it up."""
    assert MolformerXL.manifest().lane == "gpu"


def test_the_manifest_offers_freezing_the_encoder() -> None:
    spec = next(c for c in MolformerXL.manifest().conditions if c.key == "freeze_encoder")

    assert spec.type is ConditionType.BOOL
    assert spec.default is False


def test_the_checkpoint_revision_is_pinned_to_a_commit_not_a_branch() -> None:
    """`trust_remote_code=True` executes this repository's Python on the worker. An
    unpinned ref means the code that runs can change under a published Protocol
    without anything in this repository changing."""
    from daikonstudio.infrastructure.engines.molformer_xl import _REVISION

    assert re.fullmatch(r"[0-9a-f]{40}", _REVISION), _REVISION


@needs_weights
def test_regression_reports_the_shared_metric_vocabulary() -> None:
    result = MolformerXL().train(
        _context(_frame([float(i) for i in range(12)]), TaskType.REGRESSION)
    )

    assert sorted(result.metrics["y"]) == ["mae", "r2", "rmse"]
    assert result.artifact


@needs_weights
def test_classification_reports_the_shared_metric_vocabulary() -> None:
    result = MolformerXL().train(
        _context(_frame([float(i % 2) for i in range(12)]), TaskType.BINARY_CLASSIFICATION)
    )

    assert sorted(result.metrics["y"]) == ["auprc", "auroc", "balanced_accuracy", "mcc"]
    assert "accuracy" not in result.metrics["y"]


@needs_weights
def test_predict_returns_the_contracted_schema_and_dtypes() -> None:
    """An all-None uncertainty column inferred as polars Null would make this engine's
    output schema-incompatible with every other engine's."""
    trained = MolformerXL().train(
        _context(_frame([float(i) for i in range(12)]), TaskType.REGRESSION)
    )

    predictions = MolformerXL().predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": ["CCO", "CCC"]}),
            structure_column="smiles",
            artifact=trained.artifact,
            conditions={},
            target_columns=("y",),
        )
    )

    assert predictions.columns == ["row_id", "value", "uncertainty"]
    assert predictions.schema["row_id"] == pl.Int64
    assert predictions.schema["value"] == pl.Float64
    assert predictions.schema["uncertainty"] == pl.Float64
    assert predictions.height == 2


@needs_weights
def test_predictions_are_reproducible_across_calls() -> None:
    """The reason `deterministic_eval=True` is passed to `from_pretrained`.

    MoLFormer's linear attention draws random Fourier features, and
    `MolformerFeatureMap.forward` redraws them on *every* forward pass unless that
    flag is set -- so by default the same molecule scores differently on each call and
    a Protocol's predictions are not reproducible at all. This fails if the flag is
    dropped.
    """
    trained = MolformerXL().train(
        _context(_frame([float(i) for i in range(12)]), TaskType.REGRESSION)
    )
    ctx = PredictContext(
        frame=pl.DataFrame({"smiles": _SMILES}),
        structure_column="smiles",
        artifact=trained.artifact,
        conditions={},
        target_columns=("y",),
    )

    first = MolformerXL().predict(ctx)["value"].to_list()
    second = MolformerXL().predict(ctx)["value"].to_list()

    assert first == second


@needs_weights
def test_regression_predictions_come_back_in_the_targets_own_units() -> None:
    """Targets are standardized before the fit and un-standardized after it. Without
    the inverse, every prediction would come back near zero in standardized space
    while the Scorecard compared it against real assay values -- a silently wrong
    RMSE rather than an error."""
    # Deliberately far from zero: standardized predictions would sit around 0.0.
    trained = MolformerXL().train(
        _context(_frame([1000.0 + i for i in range(12)]), TaskType.REGRESSION)
    )

    predictions = MolformerXL().predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": _SMILES}),
            structure_column="smiles",
            artifact=trained.artifact,
            conditions={},
            target_columns=("y",),
        )
    )

    assert all(900.0 < value < 1100.0 for value in predictions["value"].to_list())


@needs_weights
def test_classification_predictions_are_probabilities() -> None:
    """The model emits one raw logit for both tasks; the classification path is the
    only thing that puts it through a sigmoid."""
    trained = MolformerXL().train(
        _context(_frame([float(i % 2) for i in range(12)]), TaskType.BINARY_CLASSIFICATION)
    )

    predictions = MolformerXL().predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": _SMILES}),
            structure_column="smiles",
            artifact=trained.artifact,
            conditions={},
            target_columns=("y",),
        )
    )

    assert all(0.0 <= value <= 1.0 for value in predictions["value"].to_list())


@needs_weights
def test_freezing_the_encoder_leaves_only_the_head_trainable() -> None:
    from daikonstudio.infrastructure.engines.molformer_xl import _load_backbone

    _tokenizer, model = _load_backbone(freeze_encoder=True)

    assert not any(p.requires_grad for p in model.molformer.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


@needs_weights
def test_report_is_called_once_per_epoch() -> None:
    """This callback is the only thing that can stop a fit already in flight."""
    calls: list[tuple[float, str]] = []
    ctx = TrainContext(
        frame=_frame([float(i) for i in range(12)]),
        targets={"y": TaskType.REGRESSION},
        structure_column="smiles",
        conditions=_FAST,
        seed=13,
        report=lambda fraction, phase: calls.append((fraction, phase)),
    )

    MolformerXL().train(ctx)

    assert len(calls) == 2
    assert calls[-1][0] == pytest.approx(1.0)
    # The device is named but NOT asserted to a fixed value: `accelerator="auto"`
    # legitimately resolves to mps here, cuda on a CUDA runner and cpu in CI.
    phase = calls[-1][1]
    assert phase.startswith("Training MoLFormer-XL on ")
    assert re.fullmatch(r"cpu|mps(:\d+)?|cuda(:\d+)?", phase.rsplit(" ", 1)[-1]), phase
