"""Chemprop D-MPNN: a directed message-passing neural network over the molecular graph.

The first engine that does not run wherever the API runs. Its manifest declares
`lane="gpu"`, and a deployment satisfies that by running a worker with
STUDIO_WORKER_LANE=gpu on a machine that has a GPU. Nothing here names a host.

**Every chemprop, torch and lightning import lives inside a function, never at module
scope.** That is what lets the API tier and the default-lane worker install without CUDA
and still serve this manifest, validate its conditions and render it in the engine
picker. `default_registry()` instantiates this class at import time, so the constructor
must stay import-free too.

Reproducibility here is honest, not exact. `seed_everything` fixes the Python, NumPy and
torch seeds, but GPU kernels are not bit-deterministic. The optimism-gap comparison in
`train_protocol.py` assumes the split strategy is the only variable between its two
numbers; for this engine there is also kernel-level noise. The comparison stays
directionally valid, and this paragraph is why nobody should read it as exact.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import polars as pl

from daikonstudio.application.engines.context import PredictContext, TrainContext, TrainResult
from daikonstudio.application.engines.manifest import (
    ConditionSpec,
    ConditionType,
    EngineManifest,
    TaskType,
    validate_conditions,
)
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.engines._scoring import (
    classification_metrics,
    regression_metrics,
)

_MANIFEST = EngineManifest(
    id="chemprop-dmpnn",
    version="1.0.0",
    name="Chemprop D-MPNN",
    description=(
        "A directed message-passing neural network that learns its own representation "
        "from the molecular graph instead of using a fixed fingerprint. Trains on a "
        "GPU and takes minutes to hours depending on dataset size."
    ),
    tasks=(TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION),
    conditions=(
        ConditionSpec(
            key="epochs",
            label="Training epochs",
            type=ConditionType.INTEGER,
            default=50,
            minimum=1,
            maximum=500,
            help="How many passes over the training set. More epochs fit the training "
            "data more closely, at growing risk of memorising it. 50 is a good default.",
        ),
        ConditionSpec(
            key="depth",
            label="Message passing steps",
            type=ConditionType.INTEGER,
            default=3,
            minimum=2,
            maximum=6,
            help="How far information travels across the molecule. Each step lets an "
            "atom see one bond further. 3 covers most local chemistry.",
        ),
        ConditionSpec(
            key="message_hidden_dim",
            label="Hidden size",
            type=ConditionType.INTEGER,
            default=300,
            minimum=64,
            maximum=2400,
            help="How much the network can represent about each atom. Larger needs "
            "more data to be worth it.",
        ),
        ConditionSpec(
            key="batch_size",
            label="Batch size",
            type=ConditionType.INTEGER,
            default=64,
            minimum=8,
            maximum=512,
            help="How many molecules are scored before the weights update. Lower it "
            "if training runs out of GPU memory.",
        ),
    ),
    lane="gpu",
)

# Prediction is a forward pass with no gradients, so this only trades memory against
# kernel-launch overhead. It is not the training `batch_size` condition and does not
# change any result.
_PREDICT_BATCH_SIZE = 64


def _require_chemprop() -> None:
    """Fail with a cause a human can act on, not a bare ModuleNotFoundError.

    The worker records `repr(exc)` on the Run, so this message is exactly what a
    scientist sees when their training run failed -- which makes "you are running this
    engine on a worker that cannot" the single most valuable thing it can say.
    """
    try:
        import chemprop  # noqa: F401
    except ImportError as exc:
        raise ValidationError(
            "The chemprop-dmpnn engine needs the 'gpu' extra, which this worker does "
            "not have installed. This engine declares lane 'gpu': run it on a worker "
            "started with STUDIO_WORKER_LANE=gpu (see Dockerfile.gpu), or install the "
            "extra locally with `uv sync --extra gpu`."
        ) from exc


def _datapoints(structures: list[str], targets: list[float] | None = None) -> list[Any]:
    """chemprop wants one datapoint per molecule, with `y` a 1-D array of length
    n_tasks. A scalar y silently breaks both `MoleculeDataset.t` and the target
    scaler, so the `(1,)` shape here is load-bearing rather than stylistic."""
    import numpy as np
    from chemprop.data import MoleculeDatapoint

    if targets is None:
        return [MoleculeDatapoint.from_smi(smiles) for smiles in structures]
    return [
        MoleculeDatapoint.from_smi(smiles, y=np.array([float(target)]))
        for smiles, target in zip(structures, targets, strict=True)
    ]


def _forward(trainer: Any, model: Any, dataset: Any) -> Any:
    """Run the model over a dataset and flatten to one value per molecule.

    `trainer.predict` is what puts the model in eval mode, which is what activates
    `UnscaleTransform` -- calling `model(...)` directly would silently return scaled
    values. Each batch comes back shaped (batch, 1) because n_tasks and n_targets are
    both 1; for binary classification these are already sigmoid probabilities, so
    nothing further is applied to them here.

    `shuffle=False` is not a default: `build_dataloader`'s own is `True` (its docstring
    disagrees with its signature and is wrong), and a shuffled predict loader would
    return every value against the wrong molecule.
    """
    import torch
    from chemprop.data import build_dataloader

    batches = trainer.predict(
        model, build_dataloader(dataset, batch_size=_PREDICT_BATCH_SIZE, shuffle=False)
    )
    return torch.cat(batches).cpu().numpy().reshape(-1)


class ChempropDMPNN:
    @staticmethod
    def manifest() -> EngineManifest:
        return _MANIFEST

    def train(self, ctx: TrainContext) -> TrainResult:
        _require_chemprop()

        from chemprop.data import MoleculeDataset, build_dataloader
        from chemprop.models import MPNN
        from chemprop.nn import (
            BinaryClassificationFFN,
            BondMessagePassing,
            MeanAggregation,
            RegressionFFN,
        )
        from chemprop.nn.transforms import UnscaleTransform
        from lightning import pytorch as lightning
        from lightning.pytorch.callbacks import LambdaCallback

        # Annotated `Any` rather than the declared `dict[str, object]`: every value here
        # has already been coerced to the type its ConditionSpec declares, so the `int()`
        # calls below are re-stating a guarantee, not making one.
        conditions: dict[str, Any] = validate_conditions(_MANIFEST, ctx.conditions)
        epochs = int(conditions["epochs"])
        hidden = int(conditions["message_hidden_dim"])
        depth = int(conditions["depth"])
        batch_size = int(conditions["batch_size"])
        is_classification = ctx.task is TaskType.BINARY_CLASSIFICATION

        lightning.seed_everything(ctx.seed, workers=True)

        train_rows = ctx.frame.filter(pl.col("split") == "train")
        validation_rows = ctx.frame.filter(pl.col("split") == "validation")
        test_rows = ctx.frame.filter(pl.col("split") == "test")

        train_set = MoleculeDataset(
            _datapoints(
                train_rows[ctx.structure_column].to_list(),
                train_rows[ctx.target_column].to_list(),
            )
        )
        validation_set = MoleculeDataset(
            _datapoints(
                validation_rows[ctx.structure_column].to_list(),
                validation_rows[ctx.target_column].to_list(),
            )
        )
        test_set = MoleculeDataset(_datapoints(test_rows[ctx.structure_column].to_list()))

        output_transform = None
        if not is_classification:
            # Fit the scaler on the training split only, and hand the model its
            # inverse. UnscaleTransform is a no-op in train mode by design, so the loss
            # is computed in scaled space while predictions come back in the target's
            # own unit -- which is what lets the Scorecard compare them against
            # `actual` without rescaling anything itself.
            scaler = train_set.normalize_targets()
            if len(validation_set) > 0:
                validation_set.normalize_targets(scaler)
            output_transform = UnscaleTransform.from_standard_scaler(scaler)
        # The test split is deliberately never normalised: `output_transform` is what
        # puts predictions back into real units, and scaling the truth as well would
        # cancel out silently.

        predictor = (
            BinaryClassificationFFN(input_dim=hidden)
            if is_classification
            # input_dim must equal the message-passing d_h, or the FFN's first layer
            # is built for the wrong width.
            else RegressionFFN(input_dim=hidden, output_transform=output_transform)
        )
        model = MPNN(
            message_passing=BondMessagePassing(d_h=hidden, depth=depth),
            agg=MeanAggregation(),
            predictor=predictor,
            batch_norm=True,
        )

        def _report_epoch(trainer: Any, _module: Any) -> None:
            ctx.report((trainer.current_epoch + 1) / epochs, f"training {_MANIFEST.id}")

        trainer = lightning.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=epochs,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            # The interruption point. `report` may raise RunInterrupted, which
            # propagates out of `fit` and out of `train` -- the only way to stop work
            # already running on the worker thread.
            callbacks=[LambdaCallback(on_train_epoch_end=_report_epoch)],
        )
        trainer.fit(
            model,
            build_dataloader(train_set, batch_size=batch_size, seed=ctx.seed),
            # shuffle=False: `build_dataloader` defaults it to True. Shuffling the
            # validation loader would not corrupt the loss, but it makes the number
            # depend on the seed for no reason.
            build_dataloader(validation_set, batch_size=batch_size, shuffle=False)
            if len(validation_set) > 0
            else None,
        )

        predicted = _forward(trainer, model, test_set)
        actual = test_rows[ctx.target_column].to_numpy()
        if is_classification:
            metrics = classification_metrics(
                actual,
                (predicted >= 0.5).astype(float),
                predicted,
                train_has_both_classes=train_rows[ctx.target_column].n_unique() >= 2,
            )
        else:
            metrics = regression_metrics(actual, predicted)

        # Lightning writes checkpoints to a path, so this round-trips through the
        # filesystem. `tempfile` honours TMPDIR, which is how a deployment points
        # scratch at a fast local NVMe without a setting of our own.
        with tempfile.TemporaryDirectory() as scratch:
            checkpoint = Path(scratch) / "model.ckpt"
            trainer.save_checkpoint(checkpoint)
            artifact = checkpoint.read_bytes()

        return TrainResult(artifact=artifact, metrics=metrics)

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        _require_chemprop()

        from chemprop.data import MoleculeDataset
        from chemprop.models import MPNN
        from lightning import pytorch as lightning

        with tempfile.TemporaryDirectory() as scratch:
            checkpoint = Path(scratch) / "model.ckpt"
            checkpoint.write_bytes(ctx.artifact)
            # Loaded inside the block, used outside it: the weights are in memory by
            # the time the directory is removed. MPNN.save_hyperparameters() is what
            # makes the architecture recoverable from the checkpoint alone.
            model = MPNN.load_from_checkpoint(checkpoint)

        dataset = MoleculeDataset(_datapoints(ctx.frame[ctx.structure_column].to_list()))
        trainer = lightning.Trainer(
            accelerator="auto",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )
        values = _forward(trainer, model, dataset)
        row_ids = list(range(len(values)))

        # Explicit dtypes, matching `_predict_with_tree_ensemble`. An all-None
        # uncertainty list would otherwise infer as polars' Null dtype and make this
        # engine's output schema-incompatible with the ECFP4 engines' for any caller
        # that concatenates or persists results across engines.
        #
        # ponytail: uncertainty is always null. chemprop can produce it through an MVE
        # head or an ensemble; a fabricated number would be plotted by a triage grid as
        # "the model is confident here", which is worse than an admitted absent one.
        # Upgrade path: an `uncertainty` condition selecting MveFFN for regression.
        return pl.DataFrame(
            {
                "row_id": pl.Series(row_ids, dtype=pl.Int64),
                "value": pl.Series([float(value) for value in values], dtype=pl.Float64),
                "uncertainty": pl.Series([None] * len(row_ids), dtype=pl.Float64),
            }
        )
