"""The chemprop engine's contract, on whatever device this machine has.

Skipped unless chemprop imports, because the default-lane worker and the API tier
deliberately install without it. Two epochs on twenty molecules: this is not a test of
whether a D-MPNN learns anything, it is a test that the adapter honours the Engine
contract -- metric names the Scorecard can compare, and a predict frame whose dtypes
line up with every other engine's.

Nothing here pins the device. `accelerator="auto"` resolves to MPS on Apple Silicon,
CUDA on a GPU runner and CPU in CI, and all three must satisfy the same contract.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import polars as pl
import pytest

pytest.importorskip("chemprop")

from daikonstudio.application.engines.checkpoints import Checkpoints
from daikonstudio.application.engines.context import PredictContext, RunInterrupted, TrainContext
from daikonstudio.application.engines.manifest import ConditionType, TaskType
from daikonstudio.infrastructure.engines._lightning import training_state_scope
from daikonstudio.infrastructure.engines.chemprop_dmpnn import ChempropDMPNN
from tests.fakes.blob_store import InMemoryBlobStore

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
        targets={"y": task},
        structure_column="smiles",
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

    assert sorted(result.metrics["y"]) == ["mae", "r2", "rmse"]
    assert result.artifact  # a loadable checkpoint, not an empty blob


def test_classification_reports_the_shared_metric_vocabulary() -> None:
    frame = _frame([float(i % 2) for i in range(20)])

    result = ChempropDMPNN().train(_train_context(frame, TaskType.BINARY_CLASSIFICATION))

    assert sorted(result.metrics["y"]) == ["auprc", "auroc", "balanced_accuracy", "mcc"]


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
            target_columns=("y",),
        )
    )

    assert predictions.columns == ["row_id", "value", "uncertainty", "target"]
    assert predictions.schema["row_id"] == pl.Int64
    assert predictions.schema["value"] == pl.Float64
    assert predictions.schema["uncertainty"] == pl.Float64
    assert predictions.schema["target"] == pl.String
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
            target_columns=("y",),
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
        targets={"y": TaskType.REGRESSION},
        structure_column="smiles",
        conditions=_FAST,
        seed=13,
        report=lambda fraction, phase: calls.append((fraction, phase)),
    )

    ChempropDMPNN().train(ctx)

    assert len(calls) == 2
    assert calls[-1][0] == pytest.approx(1.0)
    # The device is named but NOT asserted to a fixed value: `accelerator="auto"`
    # legitimately resolves to mps on this Mac, cuda on a CUDA runner and cpu in CI,
    # and pinning one of those here would fail on the other two machines. What must
    # hold everywhere is that a device is reported at all -- an empty or absent
    # suffix means a finished run cannot say what it ran on.
    phase = calls[-1][1]
    assert phase.startswith("Training Chemprop D-MPNN on ")
    assert re.fullmatch(r"cpu|mps(:\d+)?|cuda(:\d+)?", phase.rsplit(" ", 1)[-1]), phase


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
        n_tasks=1,
    )
    assert model.message_passing.output_dim == 2048
    # The frontend's PINNED_BY_PRETRAINED claims depth=6 for CheMeleon and disables
    # the field so the form submits it. `message_passing.depth` is set straight from
    # `checkpoint["hyper_parameters"]` above with nothing in between to override it,
    # so this is the checkpoint's own saved depth: if it ever drifted from 6, the
    # fit would still succeed at the checkpoint's real depth while the stored
    # Protocol went on recording the pinned 6 -- settings the fit never used.
    assert model.message_passing.depth == 6


def test_two_targets_train_jointly_and_predict_in_long_format() -> None:
    frame = _frame([float(i) for i in range(20)]).with_columns(
        (pl.col("y") * 2.0 + 1.0).alias("z")
    )
    engine = ChempropDMPNN()
    result = engine.train(
        TrainContext(
            frame=frame,
            targets={"y": TaskType.REGRESSION, "z": TaskType.REGRESSION},
            structure_column="smiles",
            conditions=_FAST,
            seed=13,
        )
    )
    assert list(result.metrics) == ["y", "z"]
    assert "rmse" in result.metrics["z"]
    assert result.validation_metrics is not None
    assert list(result.validation_metrics) == ["y", "z"]

    predictions = engine.predict(
        PredictContext(
            frame=frame,
            structure_column="smiles",
            artifact=result.artifact,
            conditions={},
            target_columns=("y", "z"),
        )
    )
    assert predictions.height == 2 * frame.height
    assert predictions["target"].unique().sort().to_list() == ["y", "z"]


def test_two_classification_targets_are_scored_per_column() -> None:
    frame = _frame([float(i % 2) for i in range(20)]).with_columns((1.0 - pl.col("y")).alias("z"))

    result = ChempropDMPNN().train(
        TrainContext(
            frame=frame,
            targets={"y": TaskType.BINARY_CLASSIFICATION, "z": TaskType.BINARY_CLASSIFICATION},
            structure_column="smiles",
            conditions=_FAST,
            seed=13,
        )
    )

    assert sorted(result.metrics["z"]) == ["auprc", "auroc", "balanced_accuracy", "mcc"]


def test_predict_refuses_a_target_count_the_checkpoint_was_not_trained_for() -> None:
    frame = _frame([float(i) for i in range(20)])
    trained = ChempropDMPNN().train(_train_context(frame, TaskType.REGRESSION))

    with pytest.raises(ValueError, match="predicts 1 targets, but 2 were requested"):
        ChempropDMPNN().predict(
            PredictContext(
                frame=pl.DataFrame({"smiles": ["CCO"]}),
                structure_column="smiles",
                artifact=trained.artifact,
                conditions={},
                target_columns=("y", "z"),
            )
        )


def test_chemprop_declares_that_it_learns_targets_jointly() -> None:
    assert ChempropDMPNN.manifest().supports_multitask is True


# --- positive weighting, RDKit descriptors and tuned cutoffs ------------------------


def _two_labels(tune_cutoffs: bool = False, **conditions: object) -> TrainContext:
    from tests.helpers.frames import two_binary_targets_frame

    return TrainContext(
        frame=two_binary_targets_frame(),
        targets={"a": TaskType.BINARY_CLASSIFICATION, "b": TaskType.BINARY_CLASSIFICATION},
        structure_column="smiles",
        conditions={"epochs": 2, **conditions},
        seed=1,
        tune_cutoffs=tune_cutoffs,
    )


def _predict(engine: ChempropDMPNN, frame: pl.DataFrame, artifact: bytes) -> pl.DataFrame:
    return engine.predict(
        PredictContext(
            frame=frame,
            structure_column="smiles",
            artifact=artifact,
            conditions={},
            target_columns=("a", "b"),
        )
    )


def test_descriptor_inputs_fill_from_training_rows_only():
    import numpy as np

    from daikonstudio.infrastructure.engines.chemprop_dmpnn import _descriptor_inputs

    raw = np.array([[1.0, np.nan], [3.0, np.nan], [np.nan, 5.0], [1e39, np.nan]])
    train = np.array([True, True, False, False])
    x_d, fill = _descriptor_inputs(raw, train)
    logged = np.sign(raw) * np.log1p(np.abs(raw))
    assert fill[0] == pytest.approx(np.mean(logged[:2, 0]))  # training rows only
    assert fill[1] == 0.0  # missing in every training row
    assert x_d[2, 0] == pytest.approx(fill[0])  # a non-training gap gets the training fill
    assert x_d[3, 0] == pytest.approx(np.log1p(1e39))  # heavy tail tamed, about 90
    assert np.isfinite(x_d).all()


def test_the_weighted_loss_is_torchs_pos_weight_bce():
    import torch
    from torch.nn import functional as F

    from daikonstudio.infrastructure.engines._chemprop_loss import PositiveWeightedBCELoss

    logits = torch.tensor([[2.0, -1.0], [-0.5, 0.3], [1.0, 1.0]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    loss = PositiveWeightedBCELoss([3.0, 9.0])
    expected = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=torch.tensor([3.0, 9.0])
    )
    assert loss(logits, targets).item() == pytest.approx(expected.item())


def test_chemprop_can_rebuild_the_weighted_loss_when_it_loads_a_checkpoint():
    """`MPNN._load` rebuilds a criterion that is not on CPU, from its `__dict__` by
    constructor-argument name. A `pos_weight` argument that exists only as a buffer made
    that raise `TypeError`."""
    from chemprop.models import MPNN

    from daikonstudio.infrastructure.engines._chemprop_loss import PositiveWeightedBCELoss

    rebuilt = MPNN._rebuild_metric(PositiveWeightedBCELoss([3.0, 9.0]))

    assert type(rebuilt) is PositiveWeightedBCELoss
    assert rebuilt.pos_weight.tolist() == [3.0, 9.0]


def test_the_default_build_keeps_the_stock_loss_and_no_descriptor_input():
    import torch
    from chemprop.nn.metrics import BCELoss

    from daikonstudio.infrastructure.engines.chemprop_dmpnn import _build_model

    model = _build_model(
        pretrained="none",
        weights_dir="",
        hidden=64,
        depth=2,
        is_classification=True,
        output_transform=None,
        n_tasks=1,
    )
    assert type(model.predictor.criterion) is BCELoss
    assert isinstance(model.X_d_transform, torch.nn.Identity)
    assert model.predictor.ffn[0][0].in_features == model.message_passing.output_dim


def test_a_default_classification_fit_stores_no_descriptors_loss_override_or_cutoffs():
    import io

    import torch

    result = ChempropDMPNN().train(_two_labels())

    assert result.cutoffs is None
    stored = torch.load(io.BytesIO(result.artifact), weights_only=False)
    assert "daikon_descriptors" not in stored
    assert stored["hyper_parameters"]["predictor"]["criterion"] is None  # chemprop's own BCE


def test_a_checkpoint_naming_the_stock_loss_and_no_descriptors_still_predicts():
    """What a checkpoint trained before these options looks like: its criterion is the
    stock `BCELoss` and it has no descriptor key."""
    import io

    import torch
    from chemprop.nn.metrics import BCELoss

    ctx = _two_labels()
    engine = ChempropDMPNN()
    stored = torch.load(io.BytesIO(engine.train(ctx).artifact), weights_only=False)
    stored["hyper_parameters"]["predictor"]["criterion"] = BCELoss(task_weights=[1.0, 1.0])
    legacy = io.BytesIO()
    torch.save(stored, legacy)

    assert _predict(engine, ctx.frame.head(4), legacy.getvalue()).height == 8


@pytest.mark.parametrize("pretrained", ["none", "CheMeleon"])
def test_weighting_descriptors_and_cutoffs_train_jointly_and_predict(pretrained):
    from sklearn.metrics import matthews_corrcoef

    ctx = _two_labels(
        pretrained=pretrained,
        positive_weighting="balanced",
        rdkit_descriptors=True,
        tune_cutoffs=True,
    )
    engine = ChempropDMPNN()
    result = engine.train(ctx)
    assert result.cutoffs is not None and set(result.cutoffs) <= {"a", "b"}

    out = _predict(engine, ctx.frame.head(8), result.artifact)
    assert out.height == 16
    assert out["value"].is_finite().all()

    # The reported MCC is the one at the tuned cutoff, recomputed here from the predict
    # path: a Scorecard that ignored the cutoff would score at 0.5 instead.
    test = ctx.frame.filter(pl.col("split") == "test")
    probabilities = _predict(engine, test, result.artifact)
    for column, cutoff in result.cutoffs.items():
        p = probabilities.filter(pl.col("target") == column)["value"].to_numpy()
        expected = matthews_corrcoef(test[column].to_numpy(), (p >= cutoff).astype(int))
        assert result.metrics[column]["mcc"] == pytest.approx(expected)


def test_the_cutoff_is_tuned_on_the_validation_rows(monkeypatch):
    """The MCC recompute above is self-consistent whichever partition picked the cutoff,
    so pin the source: `tuned_cutoffs` must be handed the validation rows."""
    from daikonstudio.infrastructure.engines import chemprop_dmpnn

    seen: list[tuple[int, list[str]]] = []
    original = chemprop_dmpnn.tuned_cutoffs

    def spy(columns, rows, probabilities):
        seen.append((rows.height, rows["split"].unique().to_list()))
        return original(columns, rows, probabilities)

    monkeypatch.setattr(chemprop_dmpnn, "tuned_cutoffs", spy)
    ctx = _two_labels(tune_cutoffs=True)
    ChempropDMPNN().train(ctx)

    assert seen == [(int((ctx.frame["split"] == "validation").sum()), ["validation"])]


def test_descriptors_are_scaled_on_the_training_set_once(monkeypatch):
    """chemprop's CLI normalizes the validation set too, and the model's X_d_transform
    then standardizes it a second time in eval mode. Exactly one `normalize_inputs`
    call, on the training set, with that set's own statistics."""
    import numpy as np
    from chemprop.data import MoleculeDataset

    from daikonstudio.infrastructure.chem.featurize import rdkit_descriptors
    from daikonstudio.infrastructure.engines.chemprop_dmpnn import _descriptor_inputs

    calls: list[tuple[int, str, np.ndarray]] = []
    original = MoleculeDataset.normalize_inputs

    def spy(self, key="X_d", scaler=None):
        scaler = original(self, key, scaler)
        calls.append((len(self), key, scaler.mean_))
        return scaler

    monkeypatch.setattr(MoleculeDataset, "normalize_inputs", spy)
    ctx = _two_labels(rdkit_descriptors=True)
    ChempropDMPNN().train(ctx)

    train_mask = (ctx.frame["split"] == "train").to_numpy()
    x_d, _ = _descriptor_inputs(rdkit_descriptors(ctx.frame["smiles"].to_list()), train_mask)
    assert [(n, key) for n, key, _ in calls] == [(int(train_mask.sum()), "X_d")]
    assert calls[0][2] == pytest.approx(x_d[train_mask].mean(axis=0), rel=1e-4, abs=1e-4)


def test_a_weighted_checkpoint_predicts_in_a_fresh_process(tmp_path):
    """The checkpoint pickles its criterion by qualified name, so loading it imports
    `_chemprop_loss`. In this process the class is already imported by `train`; a new
    interpreter that has not imported it is the real test."""
    import io
    import subprocess
    import sys

    import torch

    from daikonstudio.infrastructure.engines._chemprop_loss import PositiveWeightedBCELoss

    result = ChempropDMPNN().train(_two_labels(positive_weighting="sqrt_balanced"))
    stored = torch.load(io.BytesIO(result.artifact), weights_only=False)
    assert type(stored["hyper_parameters"]["predictor"]["criterion"]) is PositiveWeightedBCELoss
    artifact = tmp_path / "model.ckpt"
    artifact.write_bytes(result.artifact)
    script = (
        "import sys, polars as pl\n"
        "from daikonstudio.application.engines.context import PredictContext\n"
        "from daikonstudio.infrastructure.engines.chemprop_dmpnn import ChempropDMPNN\n"
        "loss = 'daikonstudio.infrastructure.engines._chemprop_loss'\n"
        "assert loss not in sys.modules\n"
        "out = ChempropDMPNN().predict(PredictContext(\n"
        "    frame=pl.DataFrame({'smiles': ['CCO', 'CCN']}), structure_column='smiles',\n"
        f"    artifact=open({str(artifact)!r}, 'rb').read(), conditions={{}},\n"
        "    target_columns=('a', 'b')))\n"
        "assert out.height == 4 and loss in sys.modules\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert done.returncode == 0, done.stderr


def test_a_changed_descriptor_list_is_refused_at_predict():
    import io

    import torch

    from daikonstudio.domain.shared.errors import ValidationError

    ctx = _two_labels(rdkit_descriptors=True)
    engine = ChempropDMPNN()
    result = engine.train(ctx)
    stored = torch.load(io.BytesIO(result.artifact), weights_only=False)
    assert "callbacks" not in stored  # the descriptor block's save is the artifact too
    names = stored["daikon_descriptors"]["names"]
    stored["daikon_descriptors"]["names"] = names[:-1]  # as if RDKit had dropped one
    tampered = io.BytesIO()
    torch.save(stored, tampered)

    with pytest.raises(ValidationError, match="descriptor"):
        _predict(engine, ctx.frame.head(3), tampered.getvalue())


def test_the_artifact_stores_the_model_once_without_the_training_callbacks_state() -> None:
    """`keep_best` holds a full copy of the best epoch's weights. Lightning writes any
    callback state into a checkpoint, so without the strip every Protocol would store
    the model twice."""
    import io

    import torch

    with_validation = ChempropDMPNN().train(
        _train_context(_frame([float(i) for i in range(20)]), TaskType.REGRESSION)
    )
    no_validation = ChempropDMPNN().train(
        _train_context(
            pl.DataFrame(
                {
                    "smiles": _SMILES,
                    "y": [float(i) for i in range(20)],
                    "split": ["train"] * 16 + ["test"] * 4,
                }
            ),
            TaskType.REGRESSION,
        )
    )

    stored = torch.load(
        io.BytesIO(with_validation.artifact), map_location="cpu", weights_only=False
    )
    assert "callbacks" not in stored
    assert stored["state_dict"]  # the weights are there, once
    # Without a validation set nothing selects an epoch, so no callback has state to
    # leak: that artifact is the model alone, and the two must be the same size.
    assert len(with_validation.artifact) <= 1.1 * len(no_validation.artifact)


# --- saved training state and resume ------------------------------------------------

_FOUR_EPOCHS = {**_FAST, "epochs": 4}


def _resumable_context(
    store: InMemoryBlobStore,
    report: Callable[[float, str], None],
    *,
    interval_seconds: float = 1e9,
    conditions: dict[str, object] = _FOUR_EPOCHS,
) -> TrainContext:
    checkpoints = Checkpoints(
        store, "ws/datasets/d/runs/r/checkpoints/", interval_seconds=interval_seconds
    ).scoped("model")
    return TrainContext(
        frame=_frame([float(i) for i in range(20)]),
        targets={"y": TaskType.REGRESSION},
        structure_column="smiles",
        conditions=conditions,
        seed=13,
        checkpoints=checkpoints,
        report=report,
    )


def _stopping_after_epoch_two(
    reported: list[tuple[float, str]], *, cancelled: bool = False
) -> Callable[[float, str], None]:
    def report(fraction: float, phase: str) -> None:
        reported.append((fraction, phase))
        if phase.startswith("Training") and fraction >= 2 / 4:
            raise RunInterrupted("stopped", cancelled=cancelled)

    return report


def _training_state(ctx: TrainContext) -> Checkpoints:
    scope = training_state_scope(ctx.checkpoints, "chemprop")
    assert scope is not None
    return scope


def _saved_state(ctx: TrainContext) -> bytes | None:
    return _training_state(ctx).load("training-state")


def test_a_fit_stopped_by_its_time_limit_resumes_at_the_next_epoch() -> None:
    store = InMemoryBlobStore()
    reported: list[tuple[float, str]] = []
    first = _resumable_context(store, _stopping_after_epoch_two(reported))
    with pytest.raises(RunInterrupted):
        ChempropDMPNN().train(first)
    assert _saved_state(first) is not None

    reported.clear()
    resumed = _resumable_context(store, lambda fraction, phase: reported.append((fraction, phase)))
    result = ChempropDMPNN().train(resumed)

    training = [f for f, p in reported if p.startswith("Training")]
    assert training[0] == pytest.approx(3 / 4)  # epoch 3 of 4 is the first one run
    assert any(p.startswith("Resuming") for _, p in reported)
    assert result.metrics["y"]


def test_a_periodic_save_alone_lets_a_cancelled_fit_resume() -> None:
    """A cancel saves nothing, so after one only the periodic save can hold progress:
    two epochs are done and saved when the cancel arrives at the end of the third."""
    store = InMemoryBlobStore()

    def cancel_at_the_third_epoch(fraction: float, phase: str) -> None:
        if phase.startswith("Training") and fraction >= 3 / 4:
            raise RunInterrupted("cancelled", cancelled=True)

    first = _resumable_context(store, cancel_at_the_third_epoch, interval_seconds=0)
    with pytest.raises(RunInterrupted):
        ChempropDMPNN().train(first)
    assert _saved_state(first) is not None

    reported: list[tuple[float, str]] = []
    resumed = _resumable_context(store, lambda fraction, phase: reported.append((fraction, phase)))
    ChempropDMPNN().train(resumed)

    training = [f for f, p in reported if p.startswith("Training")]
    assert training[0] == pytest.approx(3 / 4)  # epoch 3 of 4 is the first one run
    assert any(p.startswith("Resuming") for _, p in reported)


def test_a_cancelled_fit_saves_nothing() -> None:
    """The run row is already CANCELLED and the runner API refuses the write."""
    store = InMemoryBlobStore()
    ctx = _resumable_context(store, _stopping_after_epoch_two([], cancelled=True))

    with pytest.raises(RunInterrupted):
        ChempropDMPNN().train(ctx)

    assert _saved_state(ctx) is None


def test_a_store_that_cannot_save_never_fails_the_fit() -> None:
    class Failing(InMemoryBlobStore):
        def put_bytes(self, key: str, data: bytes) -> str:
            raise OSError("upload refused")

    result = ChempropDMPNN().train(
        _resumable_context(Failing(), lambda fraction, phase: None, interval_seconds=0)
    )

    assert result.metrics["y"]


def test_the_best_epoch_is_part_of_the_saved_state(tmp_path: Path) -> None:
    import math

    import torch

    store = InMemoryBlobStore()
    ctx = _resumable_context(store, _stopping_after_epoch_two([]))
    with pytest.raises(RunInterrupted):
        ChempropDMPNN().train(ctx)

    saved = tmp_path / "state.ckpt"
    saved.write_bytes(_saved_state(ctx) or b"")
    state = torch.load(saved, map_location="cpu", weights_only=False)

    assert any(
        math.isfinite(entry["best"]) and entry["best_epoch"] is not None
        for entry in state["callbacks"].values()
        if "best" in entry
    )


def test_a_resumed_fit_keeps_the_best_epoch_of_the_attempt_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without it a resumed fit forgets its best epoch and could ship a worse one: the
    callback must hold the first attempt's best before any resumed epoch runs."""
    import math

    from daikonstudio.infrastructure.engines import chemprop_dmpnn

    created: list[Any] = []
    real = chemprop_dmpnn.keep_best_epoch

    def spying(**kwargs: Any) -> Any:
        callback = real(**kwargs)
        created.append(callback)
        return callback

    monkeypatch.setattr(chemprop_dmpnn, "keep_best_epoch", spying)
    store = InMemoryBlobStore()
    with pytest.raises(RunInterrupted):
        ChempropDMPNN().train(_resumable_context(store, _stopping_after_epoch_two([])))
    saved_best = created[0].best
    assert math.isfinite(saved_best)

    at_resume: list[float] = []

    def watch(fraction: float, phase: str) -> None:
        if phase.startswith("Resuming"):  # reported before the first resumed epoch
            at_resume.append(created[1].best)

    ChempropDMPNN().train(_resumable_context(store, watch))

    assert at_resume == [saved_best]


def test_a_corrupt_but_checksum_valid_state_is_discarded() -> None:
    store = InMemoryBlobStore()
    reported: list[tuple[float, str]] = []
    ctx = _resumable_context(store, lambda fraction, phase: reported.append((fraction, phase)))
    _training_state(ctx).save("training-state", b"not a checkpoint")

    result = ChempropDMPNN().train(ctx)

    training = [f for f, p in reported if p.startswith("Training")]
    assert training[0] == pytest.approx(1 / 4)
    assert not any(p.startswith("Resuming") for _, p in reported)
    assert result.metrics["y"]


def test_best_epoch_weights_survive_a_resume_not_only_the_best_loss():
    """Restoring the best loss without the best weights would let a resumed fit ship
    its last epoch while the Scorecard believed it had kept the best one."""
    import torch

    from daikonstudio.infrastructure.engines._lightning import keep_best_epoch

    saved = keep_best_epoch()
    saved.best = -0.25
    saved.best_epoch = 3
    saved.best_state = {"w": torch.tensor([1.0, 2.0])}

    restored = keep_best_epoch()
    restored.load_state_dict(saved.state_dict())

    assert (restored.best, restored.best_epoch) == (-0.25, 3)
    assert restored.best_state is not None
    assert torch.equal(restored.best_state["w"], torch.tensor([1.0, 2.0]))


# --- ensembles ----------------------------------------------------------------------

_TWO_MODELS = {**_FAST, "ensemble_size": 2}


def _predict_y(artifact: bytes, smiles: list[str] = _SMILES) -> pl.DataFrame:
    return ChempropDMPNN().predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": smiles}),
            structure_column="smiles",
            artifact=artifact,
            conditions={},
            target_columns=("y",),
        )
    )


def test_a_single_model_reports_no_uncertainty() -> None:
    """An all-zero spread from one model would read as certainty."""
    trained = ChempropDMPNN().train(
        _train_context(_frame([float(i) for i in range(20)]), TaskType.REGRESSION)
    )
    assert _predict_y(trained.artifact)["uncertainty"].null_count() == len(_SMILES)


def test_an_ensemble_predicts_its_models_mean_and_reports_their_spread() -> None:
    import io
    from dataclasses import replace

    import numpy as np
    import torch
    from chemprop.data import MoleculeDataset

    from daikonstudio.infrastructure.engines.chemprop_dmpnn import (
        _datapoints,
        _forward,
        _model_from_bytes,
        _predict_trainer,
    )

    ctx = replace(
        _train_context(_frame([float(i) for i in range(20)]), TaskType.REGRESSION),
        conditions=_TWO_MODELS,
    )
    trained = ChempropDMPNN().train(ctx)

    stored = torch.load(io.BytesIO(trained.artifact), weights_only=False)
    members = [_model_from_bytes(data) for data in stored["daikon_ensemble"]]
    assert len(members) == 2
    # The models are stored once: torch's default pickle protocol inflates bytes 1.5x.
    assert len(trained.artifact) < 1.1 * sum(len(data) for data in stored["daikon_ensemble"])
    dataset = MoleculeDataset(_datapoints(_SMILES))
    each = np.stack([_forward(_predict_trainer(), model, dataset)[:, 0] for model in members])

    out = _predict_y(trained.artifact)
    np.testing.assert_allclose(out["value"].to_numpy(), each.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(
        out["uncertainty"].to_numpy(), each.std(axis=0), rtol=1e-4, atol=1e-6
    )

    # The Scorecard scores the ensemble, not its first model.
    test = ctx.frame.filter(pl.col("split") == "test")
    predicted = _predict_y(trained.artifact, test["smiles"].to_list())["value"].to_numpy()
    rmse = float(np.sqrt(np.mean((predicted - test["y"].to_numpy()) ** 2)))
    assert trained.metrics["y"]["rmse"] == pytest.approx(rmse, rel=1e-4)


def test_each_ensemble_member_trains_from_its_own_seed(monkeypatch) -> None:
    """Checked at the seed, not on the predictions: GPU kernels are not bit-reproducible,
    so two models trained from one seed differ slightly too, and their predictions cannot
    tell a missing seed offset from that noise."""
    from dataclasses import replace

    import chemprop.data
    from lightning import pytorch as lightning

    seeds: list[int] = []
    shuffles: list[int | None] = []
    seed_everything = lightning.seed_everything
    build_dataloader = chemprop.data.build_dataloader

    def spy_seed(seed: int, **kwargs: Any) -> int:
        seeds.append(seed)
        return seed_everything(seed, **kwargs)

    def spy_loader(dataset: Any, **kwargs: Any) -> Any:
        if kwargs.get("shuffle", True):
            shuffles.append(kwargs.get("seed"))
        return build_dataloader(dataset, **kwargs)

    monkeypatch.setattr(lightning, "seed_everything", spy_seed)
    monkeypatch.setattr(chemprop.data, "build_dataloader", spy_loader)
    ChempropDMPNN().train(
        replace(
            _train_context(_frame([float(i) for i in range(20)]), TaskType.REGRESSION),
            conditions={**_FAST, "ensemble_size": 3},
        )
    )

    assert seeds == [13, 14, 15]
    assert shuffles == [13, 14, 15]


def test_an_ensemble_reports_uncertainty_for_active_inactive_targets() -> None:
    from dataclasses import replace

    ctx = replace(
        _train_context(_frame([float(i % 2) for i in range(20)]), TaskType.BINARY_CLASSIFICATION),
        conditions=_TWO_MODELS,
    )
    out = _predict_y(ChempropDMPNN().train(ctx).artifact)

    spread = out["uncertainty"].to_numpy()
    assert out["uncertainty"].null_count() == 0
    # The spread of probabilities: never negative, never more than half the range.
    assert (spread >= 0).all() and (spread <= 0.5).all() and spread.max() > 0


def test_a_weighted_ensemble_with_descriptors_predicts_in_a_fresh_process(tmp_path):
    """Each stored model pickles its loss and descriptor transform by qualified name,
    as the one-model checkpoint does, so a new interpreter must be able to load them."""
    import subprocess
    import sys

    result = ChempropDMPNN().train(
        _two_labels(positive_weighting="balanced", rdkit_descriptors=True, ensemble_size=2)
    )
    artifact = tmp_path / "model.ckpt"
    artifact.write_bytes(result.artifact)
    script = (
        "import polars as pl\n"
        "from daikonstudio.application.engines.context import PredictContext\n"
        "from daikonstudio.infrastructure.engines.chemprop_dmpnn import ChempropDMPNN\n"
        "out = ChempropDMPNN().predict(PredictContext(\n"
        "    frame=pl.DataFrame({'smiles': ['CCO', 'CCN']}), structure_column='smiles',\n"
        f"    artifact=open({str(artifact)!r}, 'rb').read(), conditions={{}},\n"
        "    target_columns=('a', 'b')))\n"
        "assert out.height == 4, out\n"
        "assert out['uncertainty'].null_count() == 0, out\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert done.returncode == 0, done.stderr


_TWO_MODELS_FOUR_EPOCHS = {**_FOUR_EPOCHS, "ensemble_size": 2}


def test_a_stopped_ensemble_resumes_in_the_model_it_stopped_in() -> None:
    """The first model finished, so it is restored rather than refitted, and the second
    picks up at the epoch after its saved state."""
    store = InMemoryBlobStore()

    def stop_in_the_second_model(fraction: float, phase: str) -> None:
        if phase.startswith("Training") and "model 2 of 2" in phase and fraction >= 3 / 4:
            raise RunInterrupted("stopped", cancelled=False)

    # Saving every epoch, so the first model did write in-progress state to be freed.
    first = _resumable_context(
        store, stop_in_the_second_model, interval_seconds=0, conditions=_TWO_MODELS_FOUR_EPOCHS
    )
    with pytest.raises(RunInterrupted):
        ChempropDMPNN().train(first)
    # The finished model's in-progress state was freed; the stopped one's was saved.
    assert _saved_state(first) is None
    assert _training_state(first).load("fitted-model") is not None
    assert first.checkpoints is not None
    second_model = training_state_scope(first.checkpoints.scoped("member-1"), "chemprop")
    assert second_model is not None and second_model.load("training-state") is not None

    reported: list[tuple[float, str]] = []
    resumed = _resumable_context(
        store,
        lambda fraction, phase: reported.append((fraction, phase)),
        conditions=_TWO_MODELS_FOUR_EPOCHS,
    )
    result = ChempropDMPNN().train(resumed)

    phases = [phase for _, phase in reported]
    assert phases[0] == "Restored Chemprop D-MPNN model 1 of 2 from saved progress"
    assert not any(p.startswith("Training Chemprop D-MPNN model 1") for p in phases)
    assert any(p.startswith("Resuming Chemprop D-MPNN model 2 of 2 from epoch 3") for p in phases)
    training = [f for f, p in reported if p.startswith("Training")]
    assert training[0] == pytest.approx((1 + 3 / 4) / 2)  # epoch 3 of 4, second model
    assert result.metrics["y"]


def test_a_single_model_saves_no_finished_model_of_its_own() -> None:
    """Its finished fit is the whole result, which the training run saves; a second
    copy would be tens of megabytes stored for nothing."""
    ctx = _resumable_context(InMemoryBlobStore(), lambda f, p: None, interval_seconds=0)
    ChempropDMPNN().train(ctx)
    assert _training_state(ctx).load("fitted-model") is None


def test_a_saved_ensemble_member_that_no_longer_loads_is_fitted_again() -> None:
    store = InMemoryBlobStore()
    ctx = _resumable_context(store, lambda f, p: None, conditions=_TWO_MODELS_FOUR_EPOCHS)
    _training_state(ctx).save("fitted-model", b"not a model")

    reported: list[str] = []
    resumed = _resumable_context(
        store, lambda f, phase: reported.append(phase), conditions=_TWO_MODELS_FOUR_EPOCHS
    )
    ChempropDMPNN().train(resumed)

    assert not any(p.startswith("Restored") for p in reported)
    assert any(p.startswith("Training Chemprop D-MPNN model 1 of 2") for p in reported)


def test_every_epoch_is_recorded_with_validation_scores() -> None:
    """What the run page charts while chemprop trains."""
    points: list[Any] = []
    frame = _frame([float(i % 2) for i in range(20)])
    context = _train_context(frame, TaskType.BINARY_CLASSIFICATION)

    ChempropDMPNN().train(replace(context, record_epoch=points.append))

    assert [point.epoch for point in points] == [1, 2]
    assert all(point.epochs == 2 and point.val_loss is not None for point in points)
    assert set(points[-1].scores) == {"auroc", "auprc", "mcc"}
    # Kept by the default `epoch_selection`, and reported with every point.
    assert all(point.kept_by == "auprc" and point.kept_epoch in (1, 2) for point in points)
