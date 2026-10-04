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
    names = stored["daikon_descriptors"]["names"]
    stored["daikon_descriptors"]["names"] = names[:-1]  # as if RDKit had dropped one
    tampered = io.BytesIO()
    torch.save(stored, tampered)

    with pytest.raises(ValidationError, match="descriptor"):
        _predict(engine, ctx.frame.head(3), tampered.getvalue())
