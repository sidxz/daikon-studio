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

from daikonstudio.application.engines.checkpoints import Checkpoints
from daikonstudio.application.engines.context import PredictContext, RunInterrupted, TrainContext
from daikonstudio.application.engines.manifest import ConditionType, TaskType
from daikonstudio.infrastructure.engines._lightning import training_state_scope
from daikonstudio.infrastructure.engines.molformer_xl import MolformerXL
from tests.fakes.blob_store import InMemoryBlobStore

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

    assert sorted(result.metrics["y"]) == ["mae", "r2", "rmse", "spearman"]
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

    assert predictions.columns == ["row_id", "value", "uncertainty", "target"]
    assert predictions.schema["row_id"] == pl.Int64
    assert predictions.schema["value"] == pl.Float64
    assert predictions.schema["uncertainty"] == pl.Float64
    assert predictions.schema["target"] == pl.String
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

    _tokenizer, model = _load_backbone(freeze_encoder=True, num_labels=1)

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


@needs_weights
def test_two_targets_train_jointly_and_predict_in_long_format() -> None:
    frame = _frame([float(i) for i in range(12)]).with_columns(
        (pl.col("y") * 2.0 + 1.0).alias("z")
    )
    engine = MolformerXL()
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


@needs_weights
def test_each_target_is_standardized_on_its_own_scale() -> None:
    """One mean and deviation per column: a shared pair would drag the small-scale
    target toward the large one and return it in the wrong unit."""
    frame = _frame([float(i) for i in range(12)]).with_columns(
        (pl.col("y") * 0.001 + 1000.0).alias("z")
    )
    engine = MolformerXL()
    trained = engine.train(
        TrainContext(
            frame=frame,
            targets={"y": TaskType.REGRESSION, "z": TaskType.REGRESSION},
            structure_column="smiles",
            conditions=_FAST,
            seed=13,
        )
    )

    predictions = engine.predict(
        PredictContext(
            frame=frame,
            structure_column="smiles",
            artifact=trained.artifact,
            conditions={},
            target_columns=("y", "z"),
        )
    )

    z = predictions.filter(pl.col("target") == "z")["value"].to_list()
    assert all(999.0 < value < 1001.0 for value in z)


@needs_weights
def test_two_classification_targets_are_scored_per_column() -> None:
    frame = _frame([float(i % 2) for i in range(12)]).with_columns((1.0 - pl.col("y")).alias("z"))

    result = MolformerXL().train(
        TrainContext(
            frame=frame,
            targets={"y": TaskType.BINARY_CLASSIFICATION, "z": TaskType.BINARY_CLASSIFICATION},
            structure_column="smiles",
            conditions=_FAST,
            seed=13,
        )
    )

    assert sorted(result.metrics["z"]) == ["auprc", "auroc", "balanced_accuracy", "mcc"]


@needs_weights
def test_an_artifact_from_before_multi_task_still_predicts() -> None:
    """Old bundles hold a scalar mean and deviation and no `n_tasks`; every Protocol
    trained before targets could be several is stored that way."""
    import io

    import torch

    trained = MolformerXL().train(
        _context(_frame([1000.0 + i for i in range(12)]), TaskType.REGRESSION)
    )
    bundle = torch.load(io.BytesIO(trained.artifact), weights_only=True)
    (bundle["target_mean"],) = bundle.pop("target_mean")
    (bundle["target_std"],) = bundle.pop("target_std")
    del bundle["n_tasks"]
    legacy = io.BytesIO()
    torch.save(bundle, legacy)

    predictions = MolformerXL().predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": _SMILES}),
            structure_column="smiles",
            artifact=legacy.getvalue(),
            conditions={},
            target_columns=("y",),
        )
    )

    assert predictions.height == len(_SMILES)
    assert all(900.0 < value < 1100.0 for value in predictions["value"].to_list())


@needs_weights
def test_predict_refuses_a_target_count_the_bundle_was_not_trained_for() -> None:
    trained = MolformerXL().train(
        _context(_frame([float(i) for i in range(12)]), TaskType.REGRESSION)
    )

    with pytest.raises(ValueError, match="predicts 1 targets, but 2 were requested"):
        MolformerXL().predict(
            PredictContext(
                frame=pl.DataFrame({"smiles": ["CCO"]}),
                structure_column="smiles",
                artifact=trained.artifact,
                conditions={},
                target_columns=("y", "z"),
            )
        )


def test_molformer_declares_that_it_learns_targets_jointly() -> None:
    assert MolformerXL.manifest().supports_multitask is True


def test_a_structure_over_the_context_limit_is_refused_not_truncated() -> None:
    """The peptide guard. MoLFormer was pretrained on 202 tokens of SMILES, which a
    peptide passes at roughly eighteen residues, so without this the fit would run on
    a structure outside the model's distribution and report a Scorecard for it. (It
    would not even have been clipped: the tokenizer declares no `model_max_length`, so
    the old `truncation=True` never fired.) The check is in `_collate`, the one
    tokenizer call both training and predicting go through.

    A stand-in tokenizer, so this runs without the 179 MB checkpoint: one token per
    character, which is roughly what a peptide SMILES costs.
    """
    import torch

    from daikonstudio.domain.shared.errors import ValidationError
    from daikonstudio.infrastructure.engines.molformer_xl import _MAX_TOKENS, _collate

    def tokenizer(smiles: list[str], **_kwargs: object) -> dict[str, object]:
        width = max(len(text) for text in smiles)
        return {
            "input_ids": torch.zeros((len(smiles), width), dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1] * len(text) + [0] * (width - len(text)) for text in smiles]
            ),
        }

    collate = _collate(tokenizer)
    peptide = "C" * (_MAX_TOKENS + 1)

    input_ids, _mask, targets = collate([("CCO", (0.5,))])
    assert input_ids.shape == (1, 3)
    assert targets.shape == (1, 1)

    with pytest.raises(ValidationError) as caught:
        collate([("CCO", (0.5,)), (peptide, (1.0,))])

    message = str(caught.value)
    assert f"at most {_MAX_TOKENS} tokens" in message
    assert f"needs {_MAX_TOKENS + 1}" in message
    # Names the structure that is too long, and where to go instead.
    assert peptide[:60] in message
    assert "fingerprint or descriptor engine" in message


# --- positive weighting and tuned cutoffs -------------------------------------------


def test_the_module_loss_carries_the_positive_weights_and_the_default_has_none():
    import torch

    from daikonstudio.infrastructure.engines.molformer_xl import _build_module

    weighted = _build_module(
        model=torch.nn.Linear(1, 2),
        learning_rate=1e-4,
        is_classification=True,
        pos_weight=[3.0, 9.0],
    )
    assert torch.equal(weighted._loss.pos_weight, torch.tensor([3.0, 9.0]))
    default = _build_module(
        model=torch.nn.Linear(1, 2), learning_rate=1e-4, is_classification=True
    )
    assert default._loss.pos_weight is None


@needs_weights
def test_weighting_and_cutoffs_train_jointly_and_predict():
    import io
    from dataclasses import replace

    import torch
    from sklearn.metrics import matthews_corrcoef

    from tests.helpers.frames import two_binary_targets_frame

    frame = two_binary_targets_frame()
    engine = MolformerXL()
    result = engine.train(
        replace(
            TrainContext(
                frame=frame,
                targets={"a": TaskType.BINARY_CLASSIFICATION, "b": TaskType.BINARY_CLASSIFICATION},
                structure_column="smiles",
                conditions={**_FAST, "epochs": 1, "positive_weighting": "balanced"},
                seed=1,
            ),
            tune_cutoffs=True,
        )
    )
    assert result.cutoffs is not None and set(result.cutoffs) <= {"a", "b"}
    # The loss weights are a training detail: they are not in the artifact, so a weighted
    # fit and an unweighted one share a format and `predict` needs no branch.
    bundle = torch.load(io.BytesIO(result.artifact), weights_only=True)
    assert not any(key.startswith("_loss.") for key in bundle["state_dict"])
    # The artifact is a hand-built bundle, so the callbacks' state (`keep_best` holds a
    # second copy of the weights) never reaches it, unlike a Lightning checkpoint.
    assert "callbacks" not in bundle

    def predict(rows: pl.DataFrame) -> pl.DataFrame:
        return engine.predict(
            PredictContext(
                frame=rows,
                structure_column="smiles",
                artifact=result.artifact,
                conditions={},
                target_columns=("a", "b"),
            )
        )

    assert predict(frame.head(8)).height == 16
    # The reported MCC is the one at the tuned cutoff, recomputed from the predict path.
    test = frame.filter(pl.col("split") == "test")
    probabilities = predict(test)
    for column, cutoff in result.cutoffs.items():
        p = probabilities.filter(pl.col("target") == column)["value"].to_numpy()
        expected = matthews_corrcoef(test[column].to_numpy(), (p >= cutoff).astype(int))
        assert result.metrics[column]["mcc"] == pytest.approx(expected)


@needs_weights
def test_the_cutoff_is_tuned_on_the_validation_rows(monkeypatch):
    from daikonstudio.infrastructure.engines import molformer_xl
    from tests.helpers.frames import two_binary_targets_frame

    seen: list[tuple[int, list[str]]] = []
    original = molformer_xl.tuned_cutoffs

    def spy(columns, rows, probabilities):
        seen.append((rows.height, rows["split"].unique().to_list()))
        return original(columns, rows, probabilities)

    monkeypatch.setattr(molformer_xl, "tuned_cutoffs", spy)
    frame = two_binary_targets_frame()
    MolformerXL().train(
        TrainContext(
            frame=frame,
            targets={"a": TaskType.BINARY_CLASSIFICATION, "b": TaskType.BINARY_CLASSIFICATION},
            structure_column="smiles",
            conditions={**_FAST, "epochs": 1},
            seed=1,
            tune_cutoffs=True,
        )
    )

    assert seen == [(int((frame["split"] == "validation").sum()), ["validation"])]


@needs_weights
def test_a_default_classification_fit_has_no_cutoffs():
    result = MolformerXL().train(
        _context(_frame([float(i % 2) for i in range(12)]), TaskType.BINARY_CLASSIFICATION)
    )
    assert result.cutoffs is None


@needs_weights
def test_a_fit_stopped_by_its_time_limit_resumes_at_the_next_epoch() -> None:
    store = InMemoryBlobStore()
    checkpoints = Checkpoints(
        store, "ws/datasets/d/runs/r/checkpoints/", interval_seconds=1e9
    ).scoped("model")
    reported: list[tuple[float, str]] = []
    resuming = False

    def stop_after_epoch_one(fraction: float, phase: str) -> None:
        reported.append((fraction, phase))
        if phase.startswith("Training") and fraction >= 1 / 3 and not resuming:
            raise RunInterrupted("time limit", cancelled=False)

    ctx = TrainContext(
        frame=_frame([float(i) for i in range(12)]),
        targets={"y": TaskType.REGRESSION},
        structure_column="smiles",
        conditions={**_FAST, "epochs": 3},
        seed=13,
        checkpoints=checkpoints,
        report=stop_after_epoch_one,
    )
    with pytest.raises(RunInterrupted):
        MolformerXL().train(ctx)
    assert training_state_scope(checkpoints, "transformers").load("training-state") is not None

    reported.clear()
    resuming = True
    result = MolformerXL().train(ctx)

    training = [f for f, p in reported if p.startswith("Training")]
    assert training[0] == pytest.approx(2 / 3)  # epoch 2 of 3 is the first one run
    assert any(p.startswith("Resuming") for _, p in reported)
    assert result.metrics["y"]
