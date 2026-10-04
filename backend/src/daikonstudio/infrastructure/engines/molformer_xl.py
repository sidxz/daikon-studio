"""MoLFormer-XL: a transformer over the SMILES string itself.

The roster's missing architectural family. Everything else here reads a molecule as a
graph (chemprop), a fingerprint (the ECFP4 engines and the GP) or a descriptor vector;
this reads it as a sequence of tokens, which is a different inductive bias rather than
a better one. Expect no average accuracy gain -- the capability is the family.

IBM's checkpoint is Apache-2.0, 46.8M parameters, pretrained on ~1.1B molecules from
ZINC and PubChem (`arxiv:2106.09553`), and ships its own
`MolformerForSequenceClassification` head with the skip connections the paper
describes. That head is used as-is rather than reimplemented on top of the bare
encoder: it is the configuration the published numbers come from.

**Every torch and transformers import lives inside a function, never at module scope**,
for the same reason as `chemprop_dmpnn.py` -- `default_registry()` builds this class at
import time and the API tier installs without the gpu extra.

Two things about this checkpoint are load-bearing and easy to get wrong; both are
guarded below and neither fails loudly:

1. `trust_remote_code=True` is required -- the `molformer` architecture is not in
   transformers, it is Python fetched from the Hub and executed here. `_REVISION` pins
   that code to one immutable commit, so what runs on a worker is what was reviewed,
   not whatever the repo holds today.
2. `deterministic_eval=True` is required for reproducible predictions. MoLFormer uses
   linear attention with random Fourier features, and its `MolformerFeatureMap.forward`
   resamples those features on *every* forward pass unless this is set -- so the
   default configuration scores the same molecule differently on each call. With it
   set, resampling happens only in training mode and the final draw is carried in the
   `weight` buffer, which `state_dict()` saves -- so a reloaded artifact reproduces the
   numbers its own Scorecard reported.
"""

from __future__ import annotations

import io
import os
from collections.abc import Sequence
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
from daikonstudio.infrastructure.engines._lightning import keep_best_by_validation_loss
from daikonstudio.infrastructure.engines._options import POSITIVE_WEIGHTING, positive_weight
from daikonstudio.infrastructure.engines._scoring import (
    classification_by_column,
    regression_metrics,
    tuned_cutoffs,
)

#: The published checkpoint, pinned to an immutable commit. The `ibm/` namespace
#: redirects here; naming the canonical one keeps the pin meaningful.
_MODEL_ID = "ibm-research/MoLFormer-XL-both-10pct"

#: Pinned because `trust_remote_code=True` executes this repo's Python on the worker.
#: An unpinned `main` means the code that runs can change under a published Protocol
#: without anything in this repository changing. Bump it deliberately, having read the
#: diff -- that is the whole point of it being here.
_REVISION = "361063d0ad524ef77cf39b08469f6be770dc550f"

#: `max_position_embeddings` from the checkpoint's own config. Longer SMILES are
#: truncated, which is lossy but bounded -- the alternative is a shape error at the
#: embedding table. Reached only by genuinely large molecules; most drug-like SMILES
#: tokenize well under half of this.
_MAX_TOKENS = 202

_PREDICT_BATCH_SIZE = 64

_MANIFEST = EngineManifest(
    id="molformer-xl",
    version="1.0.0",
    name="MoLFormer-XL",
    description=(
        "A transformer pretrained on ~1.1 billion molecules that reads the SMILES "
        "string directly instead of the molecular graph. Learns every target of a "
        "dataset in one model. Fine-tuning takes minutes to hours on a GPU; freezing "
        "the encoder trains only the output layer, in a fraction of the time."
    ),
    tasks=(TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION),
    conditions=(
        ConditionSpec(
            key="epochs",
            label="Training epochs",
            type=ConditionType.INTEGER,
            default=10,
            minimum=1,
            maximum=200,
            help="Number of passes over the training set. A pretrained transformer "
            "adapts quickly; longer training risks degrading the pretrained "
            "representation.",
        ),
        ConditionSpec(
            key="batch_size",
            label="Batch size",
            type=ConditionType.INTEGER,
            default=32,
            minimum=8,
            maximum=256,
            help="Number of molecules per gradient update. Reduce it if training runs "
            "out of GPU memory.",
        ),
        ConditionSpec(
            key="learning_rate",
            label="Learning rate",
            type=ConditionType.NUMBER,
            default=3e-5,
            minimum=1e-6,
            maximum=1e-2,
            help="Step size of each weight update. Fine-tuning a pretrained transformer "
            "needs a small value (about 3 × 10⁻⁵); increase it toward 10⁻³ when the "  # noqa: RUF001
            "encoder is frozen, because only the output layer is trained.",
        ),
        ConditionSpec(
            key="freeze_encoder",
            label="Freeze the pretrained encoder",
            type=ConditionType.BOOL,
            default=False,
            help="Train only the output layer on top of the frozen pretrained "
            "representation. Much faster and far harder to overfit, which makes it the "
            "better choice on small assays; fine-tuning the whole model usually performs "
            "better with thousands of measurements.",
        ),
        POSITIVE_WEIGHTING,
    ),
    lane="gpu",
    supports_multitask=True,
)


def _require_transformers(weights_dir: str) -> None:
    """Fail with a cause a human can act on, and point the Hub cache at our own dir.

    `HF_HOME` is read by transformers at import time, so it is set first. Reusing
    `STUDIO_PRETRAINED_WEIGHTS_DIR` means one weights directory per deployment --
    the same one `Dockerfile.gpu` bakes CheMeleon into -- rather than a second cache
    appearing under the worker's home directory. `setdefault`, so a deployment that
    already manages `HF_HOME` keeps its own answer.
    """
    os.environ.setdefault("HF_HOME", str(os.path.expanduser(weights_dir)))
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ValidationError(
            "This runner does not have the GPU dependencies that MoLFormer-XL requires. "
            "An administrator can register a runner for the 'gpu' lane on the Runners "
            "page."
        ) from exc


def _load_backbone(*, freeze_encoder: bool, num_labels: int) -> tuple[Any, Any]:
    """The tokenizer and the classification model, both pinned to `_REVISION`.

    One logit per target (`num_labels`), read as a value for regression and through a
    sigmoid for classification. The loss is computed by the Lightning module
    rather than by passing `labels=` into the model, because the model's own
    `problem_type` inference would pick mean-squared error for a single label -- which
    is wrong for a probability. Logits and an explicit loss keep both tasks on one
    code path with nothing inferred.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        _MODEL_ID, revision=_REVISION, trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        _MODEL_ID,
        revision=_REVISION,
        trust_remote_code=True,
        num_labels=num_labels,
        # See this module's docstring: without it, every forward pass redraws the
        # linear-attention random features and the same molecule scores differently
        # on each call.
        deterministic_eval=True,
    )
    if freeze_encoder:
        # The `classifier` head stays trainable; only the pretrained trunk is pinned.
        model.molformer.requires_grad_(False)
    return tokenizer, model


def _collate(tokenizer: Any) -> Any:
    """Tokenize per batch rather than up front.

    Tokenizing a whole split eagerly would hold `n x 202` token ids in memory for the
    entire fit -- hundreds of megabytes on a large assay, most of it padding. Batching
    it also pads to the longest SMILES *in the batch* instead of the longest in the
    dataset, which is the bulk of the saving.
    """
    import torch

    def collate(batch: Sequence[tuple[str, Sequence[float]]]) -> tuple[Any, Any, Any]:
        smiles = [item[0] for item in batch]
        # (batch, n_tasks): one column per target.
        targets = torch.tensor([list(item[1]) for item in batch], dtype=torch.float32)
        encoded = tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=_MAX_TOKENS,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"], targets

    return collate


def _loader(
    examples: Sequence[tuple[str, Sequence[float]]],
    *,
    collate: Any,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
) -> Any:
    """A DataLoader over `(smiles, targets)` pairs, one target per task.

    A plain list is already a map-style dataset, so there is no Dataset subclass here
    to hold two parallel lists and re-implement `__len__`.
    """
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(
        examples,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        generator=None if seed is None else torch.Generator().manual_seed(seed),
    )


def _logits(trainer: Any, module: Any, loader: Any) -> Any:
    """The model's raw outputs over `loader`, shaped (molecules, n_tasks).

    `trainer.predict` rather than calling the module directly: it is what puts the
    module in eval mode, which is what freezes MoLFormer's random features (see this
    module's docstring). Shared by scoring and by `predict` so those two can never
    drift into running the model differently.
    """
    import torch

    batches: Any = trainer.predict(module, loader)
    return torch.cat(batches).cpu().numpy()


def _unlabelled(structures: list[str], n_tasks: int) -> list[tuple[str, tuple[float, ...]]]:
    """Structures paired with placeholder targets, for inference.

    `_collate` builds a target tensor unconditionally; zero is never read, because
    nothing computes a loss on a predict pass.
    """
    return [(smiles, (0.0,) * n_tasks) for smiles in structures]


def _to_values(logits: Any, *, is_classification: bool, target_mean: Any, target_std: Any) -> Any:
    """Raw logits to the numbers a scientist sees.

    One function, called by both the training-time scoring and `predict`, so the
    Scorecard can never describe a different transform than the one production uses --
    a divergence here would be invisible in both places.

    `target_mean` and `target_std` broadcast against the (molecules, n_tasks) logits:
    a per-column list from a current bundle, or the single scalar an older one holds.
    """
    import numpy as np

    if is_classification:
        return 1.0 / (1.0 + np.exp(-logits))
    return np.asarray(logits) * np.asarray(target_std) + np.asarray(target_mean)


def _build_module(
    *,
    model: Any,
    learning_rate: float,
    is_classification: bool,
    pos_weight: list[float] | None = None,
) -> Any:
    """The LightningModule. Defined inside a function so `lightning` imports lazily.

    `pos_weight` is one positive-class weight per target for the classification loss,
    which broadcasts it over the (batch, n_tasks) logits; None leaves the loss as it was.
    The loss only runs in training and validation, so `predict` never passes one.
    """
    import torch
    from lightning import pytorch as lightning

    class _MolformerModule(lightning.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = model
            self._loss = (
                torch.nn.BCEWithLogitsLoss(
                    pos_weight=None if pos_weight is None else torch.tensor(pos_weight)
                )
                if is_classification
                else torch.nn.MSELoss()
            )

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            return output.logits

        def training_step(self, batch: Any, _index: int) -> Any:
            input_ids, attention_mask, targets = batch
            loss = self._loss(self(input_ids, attention_mask), targets)
            self.log("train_loss", loss, batch_size=len(targets))
            return loss

        def validation_step(self, batch: Any, _index: int) -> Any:
            input_ids, attention_mask, targets = batch
            loss = self._loss(self(input_ids, attention_mask), targets)
            # The key `keep_best_by_validation_loss` reads. `batch_size` is explicit
            # because Lightning cannot infer it from a tuple batch and warns per step.
            self.log("val_loss", loss, batch_size=len(targets))
            return loss

        def predict_step(self, batch: Any, _index: int) -> Any:
            input_ids, attention_mask, _targets = batch
            return self(input_ids, attention_mask)

        def configure_optimizers(self) -> Any:
            # Only the parameters that are actually learning: under `freeze_encoder`
            # handing AdamW the frozen trunk would allocate optimizer state for 46M
            # parameters that never receive a gradient.
            trainable = [p for p in self.parameters() if p.requires_grad]
            return torch.optim.AdamW(trainable, lr=learning_rate)

    return _MolformerModule()


def _standardize(targets: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    """Per-target mean and standard deviation of the training values, for regression.

    A transformer fine-tuned with mean-squared error against raw assay values --
    percent inhibition, IC50 in micromolar, log units -- has its gradients scaled by
    whatever unit the column happens to use. Standardizing on the *training* split
    only, and inverting at predict time, is the same thing chemprop's
    `UnscaleTransform` does, so predictions come back in the target's own unit and the
    Scorecard compares them against `actual` without rescaling anything itself.

    A zero standard deviation (every training value identical) would divide by zero;
    1.0 leaves the values as they are, which is the honest answer for a constant column.
    Each column gets its own, so a target on a small scale is not dragged toward one on
    a large scale.
    """
    import numpy as np

    array = np.asarray(targets, dtype=float)
    deviation = array.std(axis=0)
    return array.mean(axis=0).tolist(), np.where(deviation > 0.0, deviation, 1.0).tolist()


class MolformerXL:
    @staticmethod
    def manifest() -> EngineManifest:
        return _MANIFEST

    def train(self, ctx: TrainContext) -> TrainResult:
        from daikonstudio.settings import Settings

        _require_transformers(Settings().pretrained_weights_dir)

        import torch
        from lightning import pytorch as lightning
        from lightning.pytorch.callbacks import LambdaCallback

        conditions: dict[str, Any] = validate_conditions(_MANIFEST, ctx.conditions)
        epochs = int(conditions["epochs"])
        batch_size = int(conditions["batch_size"])
        learning_rate = float(conditions["learning_rate"])
        freeze_encoder = bool(conditions["freeze_encoder"])
        weighting = str(conditions["positive_weighting"])
        columns = ctx.target_columns
        # One task for the whole fit: a joint engine is only ever handed a dataset
        # whose targets share a kind (`joint_kind_error`, at enqueue).
        is_classification = ctx.task is TaskType.BINARY_CLASSIFICATION

        lightning.seed_everything(ctx.seed, workers=True)

        train_rows = ctx.frame.filter(pl.col("split") == "train")
        validation_rows = ctx.frame.filter(pl.col("split") == "validation")
        test_rows = ctx.frame.filter(pl.col("split") == "test")

        # Regression targets are standardized; classification targets are already
        # 0/1 and BCEWithLogitsLoss expects them that way.
        target_mean, target_std = (
            ([0.0] * len(columns), [1.0] * len(columns))
            if is_classification
            else _standardize(train_rows.select(columns).rows())
        )

        def examples(rows: pl.DataFrame) -> list[tuple[str, tuple[float, ...]]]:
            structures = rows[ctx.structure_column].to_list()
            return [
                (
                    smiles,
                    tuple(
                        (float(value) - mean) / std
                        for value, mean, std in zip(row, target_mean, target_std, strict=True)
                    ),
                )
                for smiles, row in zip(structures, rows.select(columns).rows(), strict=True)
            ]

        tokenizer, model = _load_backbone(freeze_encoder=freeze_encoder, num_labels=len(columns))
        pos_weight = (
            [
                positive_weight(train_rows[column].to_numpy(), weighting) or 1.0
                for column in columns
            ]
            if is_classification and weighting != "none"
            else None
        )
        module = _build_module(
            model=model,
            learning_rate=learning_rate,
            is_classification=is_classification,
            pos_weight=pos_weight,
        )
        collate = _collate(tokenizer)

        train_loader = _loader(
            examples(train_rows),
            collate=collate,
            batch_size=batch_size,
            shuffle=True,
            seed=ctx.seed,
        )
        # Shuffling a validation loader would not corrupt the loss, but it makes the
        # number depend on the seed for no reason.
        validation_loader = (
            _loader(examples(validation_rows), collate=collate, batch_size=batch_size)
            if validation_rows.height > 0
            else None
        )

        def _report_epoch(trainer: Any, _module: Any) -> None:
            # Naming the device is not decoration -- see chemprop_dmpnn.py. This is
            # also the interruption point: `report` may raise RunInterrupted, which
            # propagates out of `fit` and is the only way to stop a fit already
            # running on the worker thread.
            ctx.report(
                (trainer.current_epoch + 1) / epochs,
                f"Training {_MANIFEST.name} on {trainer.strategy.root_device}",
            )

        keep_best = keep_best_by_validation_loss()
        callbacks: list[Any] = [LambdaCallback(on_train_epoch_end=_report_epoch)]
        if validation_loader is not None:
            callbacks.append(keep_best)

        trainer = lightning.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=epochs,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            callbacks=callbacks,
        )
        trainer.fit(module, train_loader, validation_loader)

        # Restore the selected epoch before anything is scored or saved, so the
        # numbers on the Scorecard and the weights in the artifact are one model.
        if keep_best.best_state is not None:
            module.load_state_dict(keep_best.best_state)

        def infer(rows: pl.DataFrame) -> Any:
            loader = _loader(
                _unlabelled(rows[ctx.structure_column].to_list(), len(columns)),
                collate=collate,
                batch_size=_PREDICT_BATCH_SIZE,
            )
            return _to_values(
                _logits(trainer, module, loader),
                is_classification=is_classification,
                target_mean=target_mean,
                target_std=target_std,
            )

        # The validation partition is inferred once: it picks the cutoffs, then is scored
        # at them. Test is scored at those cutoffs and never sees them chosen.
        validation_values = infer(validation_rows) if validation_rows.height > 0 else None
        cutoffs: dict[str, float] = {}
        if ctx.tune_cutoffs and is_classification and validation_values is not None:
            cutoffs = tuned_cutoffs(columns, validation_rows, validation_values)

        def score(rows: pl.DataFrame, values: Any) -> dict[str, dict[str, float]]:
            if is_classification:
                return classification_by_column(columns, rows, values, cutoffs, train_rows)
            return {
                column: regression_metrics(rows[column].to_numpy(), values[:, index])
                for index, column in enumerate(columns)
            }

        metrics = score(test_rows, infer(test_rows))
        validation_metrics = (
            None if validation_values is None else score(validation_rows, validation_values)
        )

        # The whole fine-tuned model, not just the head. Under `freeze_encoder` most
        # of these ~190 MB duplicate the public checkpoint, which is wasteful -- and
        # deliberate: a published Protocol then keeps predicting even if the Hub is
        # unreachable or the repository is withdrawn, and predict needs no branch for
        # which parts were trained.
        #
        # ponytail: ~190 MB per Protocol regardless of what was trained. Upgrade path
        # if storage bites: save only parameters whose `requires_grad` was true, and
        # rebuild the rest from `_REVISION`.
        buffer = io.BytesIO()
        torch.save(
            {
                # Without the loss: a weighted fit's `pos_weight` is a buffer of it, and
                # `predict` builds its module with no loss weights to load it into.
                "state_dict": {
                    key: value
                    for key, value in module.state_dict().items()
                    if not key.startswith("_loss.")
                },
                "is_classification": is_classification,
                "target_mean": target_mean,
                "target_std": target_std,
                "n_tasks": len(columns),
                "revision": _REVISION,
            },
            buffer,
        )

        return TrainResult(
            artifact=buffer.getvalue(),
            metrics=metrics,
            validation_metrics=validation_metrics,
            cutoffs=cutoffs or None,
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        from daikonstudio.settings import Settings

        _require_transformers(Settings().pretrained_weights_dir)

        import torch
        from lightning import pytorch as lightning

        # weights_only=True: the artifact holds tensors, plain scalars and lists of them,
        # so there is no reason to allow the pickle in it to execute anything.
        bundle = torch.load(io.BytesIO(ctx.artifact), weights_only=True)
        is_classification = bool(bundle["is_classification"])
        # An artifact from before targets could be several has no `n_tasks` and a scalar
        # mean and deviation; one task, and `_to_values` broadcasts the scalars.
        n_tasks = int(bundle.get("n_tasks", 1))

        # `freeze_encoder=False` here regardless of how it was trained: the flag only
        # controls which parameters receive gradients, and nothing is training now.
        tokenizer, model = _load_backbone(freeze_encoder=False, num_labels=n_tasks)
        module = _build_module(
            model=model, learning_rate=1e-4, is_classification=is_classification
        )
        # Restores the fine-tuned weights *and* the linear-attention `weight` buffer,
        # which is what makes these predictions identical to the ones the Scorecard
        # was built from.
        module.load_state_dict(bundle["state_dict"])

        loader = _loader(
            _unlabelled(ctx.frame[ctx.structure_column].to_list(), n_tasks),
            collate=_collate(tokenizer),
            batch_size=_PREDICT_BATCH_SIZE,
        )
        trainer = lightning.Trainer(
            accelerator="auto",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )
        values = _to_values(
            _logits(trainer, module, loader),
            is_classification=is_classification,
            target_mean=bundle["target_mean"],
            target_std=bundle["target_std"],
        )
        # As for chemprop: a bundle with fewer tasks than requested would raise an
        # IndexError below, and one with more would drop a task silently.
        if values.shape[1] != len(ctx.target_columns):
            raise ValueError(
                f"The model predicts {values.shape[1]} targets, but {len(ctx.target_columns)} "
                "were requested."
            )

        # Explicit dtypes, matching every other engine: an all-None uncertainty list
        # would infer as polars' Null dtype and make this engine's output
        # schema-incompatible with theirs.
        #
        # ponytail: uncertainty is always null, as for chemprop. A transformer can
        # produce one through MC-dropout or a deep ensemble; a fabricated number would
        # be plotted by the triage grid as "the model is confident here", which is
        # worse than an admitted absent one.
        #
        # Long format, one block per target: a joint engine is not wrapped in `FanOut`,
        # so it tags its own rows.
        row_ids = pl.Series(range(values.shape[0]), dtype=pl.Int64)
        return pl.concat(
            [
                pl.DataFrame(
                    {
                        "row_id": row_ids,
                        "value": pl.Series(
                            [float(value) for value in values[:, index]], dtype=pl.Float64
                        ),
                        "uncertainty": pl.Series([None] * values.shape[0], dtype=pl.Float64),
                        "target": pl.Series([column] * values.shape[0], dtype=pl.String),
                    }
                )
                for index, column in enumerate(ctx.target_columns)
            ]
        )
