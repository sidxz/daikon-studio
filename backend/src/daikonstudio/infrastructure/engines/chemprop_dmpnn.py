"""Chemprop D-MPNN: a directed message-passing neural network over the molecular graph.

The first engine that does not run wherever the API runs. Its manifest declares
`lane="gpu"`, and a deployment satisfies that by running a registered runner for
the "gpu" lane on a machine that has a GPU. Runners are registered in the UI or
seeded locally with `make seed-runners`. Nothing here names a host.

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

import io
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from daikonstudio.application.engines.checkpoints import TRAINING_STATE, Checkpoints
from daikonstudio.application.engines.context import PredictContext, TrainContext, TrainResult
from daikonstudio.application.engines.manifest import (
    ENSEMBLE_SIZE,
    ConditionSpec,
    ConditionType,
    EngineManifest,
    TaskType,
    validate_conditions,
)
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.engines._lightning import (
    keep_best_epoch,
    record_epochs,
    save_training_state,
    saved_training_state,
    training_state_scope,
)
from daikonstudio.infrastructure.engines._options import (
    EPOCH_SELECTION,
    POSITIVE_WEIGHTING,
    RDKIT_DESCRIPTORS,
    positive_weight,
)
from daikonstudio.infrastructure.engines._scoring import (
    _require_matching_features,
    classification_by_column,
    regression_metrics,
    tuned_cutoffs,
)

_MANIFEST = EngineManifest(
    id="chemprop-dmpnn",
    version="1.0.0",
    name="Chemprop D-MPNN",
    description=(
        "A directed message-passing neural network that learns its own representation "
        "from the molecular graph instead of using a fixed fingerprint. Learns every "
        "target of a dataset in one model. Trains on a GPU and takes minutes to hours "
        "depending on dataset size."
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
            help="Number of passes over the training set. More epochs fit the training "
            "data more closely, with increasing risk of overfitting.",
        ),
        ConditionSpec(
            key="depth",
            label="Message-passing steps",
            type=ConditionType.INTEGER,
            default=3,
            minimum=2,
            maximum=6,
            help="How far information travels across the molecule. Each step lets an "
            "atom see one bond further. 3 covers most local chemistry.",
        ),
        ConditionSpec(
            key="message_hidden_dim",
            label="Hidden dimension",
            type=ConditionType.INTEGER,
            default=300,
            minimum=64,
            maximum=2400,
            help="Dimension of the learned message vectors. Larger values need more "
            "training data to be beneficial.",
        ),
        ConditionSpec(
            # The key MoLFormer's setting uses, so a sweep across both engines varies
            # one setting. Here it is the peak of chemprop's warm-up-then-decay
            # schedule, hence the label.
            key="learning_rate",
            label="Peak learning rate",
            type=ConditionType.NUMBER,
            default=1e-3,
            minimum=1e-6,
            maximum=1e-2,
            help="The largest step size of each weight update. Training starts at a tenth "
            "of it, rises to it over the first two epochs, then decays back to a tenth by "
            "the last epoch. The default, 0.001, is chemprop's own. Lower it if validation "
            "loss jumps from one epoch to the next.",
        ),
        ConditionSpec(
            key="batch_size",
            label="Batch size",
            type=ConditionType.INTEGER,
            # 128, not chemprop's 64: on the 10k nuisance set on an RTX 6000 Ada it ran
            # 1.6x faster per epoch with validation PR AUC 0.509 against 0.496 (30
            # epochs, two seeds). 256 matched it but kept epoch 27 of 30, still
            # improving; 512 was no faster than 256 and scored 0.495.
            default=128,
            minimum=8,
            maximum=512,
            help="Number of molecules per gradient update. Reduce it if training runs "
            "out of GPU memory.",
        ),
        ConditionSpec(
            key="pretrained",
            label="Pretrained weights",
            type=ConditionType.ENUM,
            default="none",
            options=("none", "CheMeleon"),
            help="Start from a foundation model's learned representation instead of "
            "random weights. CheMeleon was pretrained on ~1M PubChem molecules against "
            "classical descriptors; it fixes the hidden size at 2048 and the message "
            "passing steps at 6, so those two settings are ignored when it is selected.",
        ),
        POSITIVE_WEIGHTING,
        EPOCH_SELECTION,
        RDKIT_DESCRIPTORS,
        ConditionSpec(
            key=ENSEMBLE_SIZE,
            label="Ensemble size",
            type=ConditionType.INTEGER,
            default=1,
            minimum=1,
            maximum=10,
            help="Trains this many models, each from a different random initialization, "
            "and predicts with their average. The spread of their predictions is reported "
            "as the uncertainty, for active/inactive targets as well as continuous ones. "
            "Each model is a full training run, so five models take about five times as "
            "long. With pretrained weights every model starts from the same encoder, so "
            "the models agree more closely.",
        ),
    ),
    lane="gpu",
    supports_multitask=True,
)

logger = logging.getLogger(__name__)

# Prediction is a forward pass with no gradients, so this only trades memory against
# kernel-launch overhead. It is not the training `batch_size` condition and does not
# change any result.
_PREDICT_BATCH_SIZE = 64

# The artifact key holding an ensemble's models, each in `_model_bytes`' layout. A
# one-model artifact is a plain Lightning checkpoint, exactly as before ensembles.
_ENSEMBLE = "daikon_ensemble"
# Where a fitted ensemble member is saved, so a stopped run resumes at the next one.
_FITTED_MODEL = "fitted-model"


def _require_chemprop() -> None:
    """Fail with a cause a human can act on, not a bare ModuleNotFoundError.

    The worker records `user_facing_error(exc)` on the Run -- a DomainError's message
    passes through verbatim -- so this message is exactly what a
    scientist sees when their training run failed -- which makes "you are running this
    engine on a worker that cannot" the single most valuable thing it can say.
    """
    try:
        import chemprop  # noqa: F401
    except ImportError as exc:
        raise ValidationError(
            "This runner does not have the GPU dependencies that Chemprop D-MPNN "
            "requires. An administrator can register a runner for the 'gpu' lane on the "
            "Runners page."
        ) from exc


def _datapoints(
    structures: list[str],
    targets: Sequence[Sequence[float]] | None = None,
    x_d: Sequence[Any] | None = None,
) -> list[Any]:
    """chemprop wants one datapoint per molecule, with `y` a 1-D array of length
    n_tasks. A scalar y silently breaks both `MoleculeDataset.t` and the target
    scaler, so the 1-D shape here is load-bearing rather than stylistic, even
    when there is only one task.

    `x_d` is one row of extra molecule-level features (the RDKit descriptors) per
    structure, or None for none."""
    import numpy as np
    from chemprop.data import MoleculeDatapoint

    return [
        MoleculeDatapoint.from_smi(
            smiles,
            y=None if targets is None else np.array([float(value) for value in targets[index]]),
            x_d=None if x_d is None else x_d[index],
        )
        for index, smiles in enumerate(structures)
    ]


def _signed_log(raw: Any) -> Any:
    import numpy as np

    return np.sign(raw) * np.log1p(np.abs(raw))


def _descriptor_inputs(raw: Any, train_mask: Any) -> tuple[Any, Any]:
    """RDKit descriptors made fit for a neural network, fitted on training rows only.

    A signed log first -- order-preserving and parameter-free -- because RDKit
    descriptors are heavy-tailed (`Ipc` reaches 1e39) and a plain standardization would
    let one outlier squash every other molecule to zero. Then each missing value is
    filled with that descriptor's training-row mean (0.0 for one missing in every
    training row). Standardization is the third step and is chemprop's own:
    `normalize_inputs("X_d")` on the training set, installed as the model's
    `X_d_transform`. Returns the filled matrix and the fill values; predict reapplies
    both from the checkpoint.
    """
    import numpy as np

    logged = _signed_log(raw)
    train = logged[train_mask]
    observed = ~np.isnan(train)
    counts = observed.sum(axis=0)
    sums = np.where(observed, train, 0.0).sum(axis=0)
    fill = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return np.where(np.isnan(logged), fill, logged), fill


def _forward(trainer: Any, model: Any, dataset: Any) -> Any:
    """Run the model over a dataset and return (molecules, tasks).

    `trainer.predict` is what puts the model in eval mode, which is what activates
    `UnscaleTransform` -- calling `model(...)` directly would silently return scaled
    values. Each batch comes back shaped (batch, n_tasks); for binary classification
    these are already sigmoid probabilities, so nothing further is applied to them here.

    `shuffle=False` is not a default: `build_dataloader`'s own is `True` (its docstring
    disagrees with its signature and is wrong), and a shuffled predict loader would
    return every value against the wrong molecule.
    """
    import torch
    from chemprop.data import build_dataloader

    batches = trainer.predict(
        model, build_dataloader(dataset, batch_size=_PREDICT_BATCH_SIZE, shuffle=False)
    )
    return torch.cat(batches).cpu().numpy().reshape(len(dataset), -1)


def _ensemble_forward(trainer: Any, models: Sequence[Any], dataset: Any) -> tuple[Any, Any]:
    """The models' mean prediction and their spread, each (molecules, tasks).

    The spread is the population standard deviation across the models, in the
    prediction's own unit: a probability for a binary target, the target's unit for a
    continuous one. For one model it is all zeros, which `predict` reports as no
    uncertainty rather than as certainty.

    ponytail: each model's pass featurizes every molecule again (chemprop builds graphs
    on the fly), so featurization costs N times over. Caching the graphs would hold the
    whole upload in memory, since predict is not chunked; run the models batch by batch
    if prediction time on large libraries matters.
    """
    import numpy as np

    values = np.stack([_forward(trainer, model, dataset) for model in models])
    return values.mean(axis=0), values.std(axis=0)


def _predict_trainer() -> Any:
    from lightning import pytorch as lightning

    return lightning.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )


def _validation_scores(module: Any, batch: Any) -> tuple[Any, Any]:
    """A validation batch's predicted probabilities and labels, each (rows, targets),
    unlabelled entries NaN: what `keep_best_epoch` scores an epoch by."""
    molecules, atom_features, descriptors, targets, *_ = batch
    return module(molecules, atom_features, descriptors), targets


def _model_bytes(model: Any) -> bytes:
    """One fitted model in chemprop's own `save_model` layout: hyperparameters and
    weights, without the optimizer state a Lightning checkpoint also carries. This is
    what an ensemble stores per model, and `MPNN.load_from_file` reads it back."""
    import torch

    buffer = io.BytesIO()
    torch.save({"hyper_parameters": model.hparams, "state_dict": model.state_dict()}, buffer)
    return buffer.getvalue()


def _model_from_bytes(data: bytes) -> Any:
    from chemprop.models import MPNN

    # chemprop's loader unpickles (weights_only=False) -- the same trust boundary as the
    # one-model artifact in `predict`: these bytes are only ever our own `train`'s.
    return MPNN.load_from_file(io.BytesIO(data), map_location="cpu")


def _member_checkpoints(checkpoints: Checkpoints | None, index: int) -> Checkpoints | None:
    """Where ensemble member `index` saves its progress. The first saves where a single
    model always has, so a one-model fit -- including a run stopped before ensembles
    existed -- resumes exactly as it did."""
    if checkpoints is None or index == 0:
        return checkpoints
    return checkpoints.scoped(f"member-{index}")


def _restored_model(state: Checkpoints) -> Any:
    """An ensemble member an earlier attempt finished, or None to fit it. A save that no
    longer loads is fitted again rather than failing the run."""
    data = state.load(_FITTED_MODEL)
    if data is None:
        return None
    try:
        return _model_from_bytes(data)
    except Exception:
        logger.warning(
            "Could not restore a fitted ensemble member; fitting it again", exc_info=True
        )
        return None


def _build_model(
    *,
    pretrained: str,
    weights_dir: str,
    hidden: int,
    depth: int,
    is_classification: bool,
    output_transform: Any,
    n_tasks: int,
    criterion: Any = None,
    n_descriptors: int = 0,
    x_d_transform: Any = None,
    learning_rate: float = 1e-3,
) -> Any:
    """The network, before any data touches it.

    Under `pretrained`, the architecture comes from the checkpoint's own saved
    hyperparameters rather than from the conditions -- a request that bypassed
    the form must not be able to build a network the weights do not fit. The
    form pins and disables the two inert conditions so the stored record still
    matches what ran; this function is what makes that safe rather than trusted.

    `criterion` replaces the classification head's stock BCE (None keeps it);
    `n_descriptors` widens the predictor's input by that many molecule-level features
    concatenated onto the graph embedding, and `x_d_transform` standardizes them.
    `learning_rate` is the schedule's peak; it starts and ends at a tenth of that, the
    ratio of chemprop's own defaults (1e-4, 1e-3, 1e-4), so 1e-3 trains exactly as an
    unset one did.
    """
    import torch
    from chemprop.models import MPNN
    from chemprop.nn import (
        BinaryClassificationFFN,
        BondMessagePassing,
        MeanAggregation,
        RegressionFFN,
    )
    from chemprop.nn.metrics import (
        MAE,
        RMSE,
        BinaryAUPRC,
        BinaryAUROC,
        BinaryMCCMetric,
        R2Score,
    )

    from daikonstudio.infrastructure.engines._pretrained import weights_path

    if pretrained == "none":
        message_passing = BondMessagePassing(d_h=hidden, depth=depth)
        # An untrained encoder benefits from batch norm on the graph embedding.
        batch_norm = True
    else:
        # NOT MPNN.load_from_checkpoint: this file is not a Lightning checkpoint.
        # It holds exactly two keys -- `hyper_parameters` and `state_dict` -- for
        # the message-passing block alone, with no predictor head, and
        # load_from_checkpoint raises KeyError('metrics') on it.
        checkpoint = torch.load(weights_path(pretrained, weights_dir), weights_only=True)
        message_passing = BondMessagePassing(**checkpoint["hyper_parameters"])
        message_passing.load_state_dict(checkpoint["state_dict"])
        # False to match chemprop's own chemeleon_foundation_finetuning notebook.
        batch_norm = False

    # Must equal the message passing's output width, which under a pretrained
    # encoder is the checkpoint's d_h (2048 for CheMeleon), not `hidden`, plus any
    # descriptors appended to it.
    input_dim = message_passing.output_dim + n_descriptors
    predictor = (
        BinaryClassificationFFN(input_dim=input_dim, n_tasks=n_tasks, criterion=criterion)
        if is_classification
        else RegressionFFN(input_dim=input_dim, n_tasks=n_tasks, output_transform=output_transform)
    )
    return MPNN(
        message_passing=message_passing,
        # CheMeleon requires mean aggregation, which is also what this engine
        # has always used -- so there is nothing to branch on.
        agg=MeanAggregation(),
        predictor=predictor,
        batch_norm=batch_norm,
        X_d_transform=x_d_transform,
        init_lr=learning_rate / 10,
        max_lr=learning_rate,
        final_lr=learning_rate / 10,
        # Validation scores per epoch, for the run page's live charts (see
        # `_lightning.record_epochs`). Logging only: `keep_best_epoch` selects the epoch,
        # from scores it averages over targets itself (these pool them) or from the
        # validation loss chemprop appends after these as `val_loss`.
        metrics=[BinaryAUROC(), BinaryAUPRC(), BinaryMCCMetric()]
        if is_classification
        else [RMSE(), MAE(), R2Score()],
    )


class ChempropDMPNN:
    @staticmethod
    def manifest() -> EngineManifest:
        return _MANIFEST

    def train(self, ctx: TrainContext) -> TrainResult:
        _require_chemprop()

        import torch
        from chemprop.data import MoleculeDataset, build_dataloader
        from chemprop.nn.transforms import ScaleTransform, UnscaleTransform
        from lightning import pytorch as lightning
        from lightning.pytorch.callbacks import LambdaCallback

        from daikonstudio.settings import Settings

        # Annotated `Any` rather than the declared `dict[str, object]`: every value here
        # has already been coerced to the type its ConditionSpec declares, so the `int()`
        # calls below are re-stating a guarantee, not making one.
        conditions: dict[str, Any] = validate_conditions(_MANIFEST, ctx.conditions)
        epochs = int(conditions["epochs"])
        hidden = int(conditions["message_hidden_dim"])
        depth = int(conditions["depth"])
        batch_size = int(conditions["batch_size"])
        learning_rate = float(conditions["learning_rate"])
        pretrained = str(conditions["pretrained"])
        use_descriptors = bool(conditions["rdkit_descriptors"])
        weighting = str(conditions["positive_weighting"])
        selection = str(conditions["epoch_selection"])
        members = int(conditions[ENSEMBLE_SIZE])
        columns = ctx.target_columns
        # One task for the whole fit: a joint engine is only ever handed a dataset
        # whose targets share a kind (`joint_kind_error`, at enqueue).
        is_classification = ctx.task is TaskType.BINARY_CLASSIFICATION

        train_rows = ctx.frame.filter(pl.col("split") == "train")
        validation_rows = ctx.frame.filter(pl.col("split") == "validation")
        test_rows = ctx.frame.filter(pl.col("split") == "test")

        # RDKit descriptors for every row, fitted on the training rows only; each
        # partition below takes its own rows, in frame order like the `filter`s above.
        x_d_all = fill = None
        if use_descriptors:
            from daikonstudio.infrastructure.chem.featurize import rdkit_descriptors

            x_d_all, fill = _descriptor_inputs(
                rdkit_descriptors(ctx.frame[ctx.structure_column].to_list()),
                (ctx.frame["split"] == "train").to_numpy(),
            )

        def x_d_for(partition: str) -> Any:
            return (
                None if x_d_all is None else x_d_all[(ctx.frame["split"] == partition).to_numpy()]
            )

        train_set = MoleculeDataset(
            _datapoints(
                train_rows[ctx.structure_column].to_list(),
                train_rows.select(columns).rows(),
                x_d_for("train"),
            )
        )
        validation_set = MoleculeDataset(
            _datapoints(
                validation_rows[ctx.structure_column].to_list(),
                validation_rows.select(columns).rows(),
                x_d_for("validation"),
            )
        )
        target_scaler = None
        if not is_classification:
            # Fit the scaler on the training split only, and hand the model its
            # inverse. UnscaleTransform is a no-op in train mode by design, so the loss
            # is computed in scaled space while predictions come back in the target's
            # own unit -- which is what lets the Scorecard compare them against
            # `actual` without rescaling anything itself.
            # One scaler per target column, so each task is standardized on its own scale.
            target_scaler = train_set.normalize_targets()
            if len(validation_set) > 0:
                validation_set.normalize_targets(target_scaler)
        # The test split is deliberately never normalised: `output_transform` is what
        # puts predictions back into real units, and scaling the truth as well would
        # cancel out silently.

        x_d_scaler = None
        if use_descriptors:
            # Scale the TRAINING set only. ScaleTransform is a no-op in train mode and
            # standardizes in eval mode, and Lightning validates in eval mode -- so
            # normalizing the validation set too (as chemprop's own CLI does) would
            # standardize it twice. Validation, test and predict inputs stay raw (signed
            # log + fill) and the model scales them once.
            x_d_scaler = train_set.normalize_inputs("X_d")

        # Featurize each molecule once for the whole fit. Uncached, chemprop rebuilds a
        # molecule's graph every time a batch draws it, on the training process's one
        # thread, every epoch and for every ensemble model. Chemprop's own CLI caches by
        # default. After the normalization above, as the CLI does; descriptors are not
        # part of the cached graph.
        # ponytail: held in memory, about 17 KB a molecule (7 GB at 400k). Featurize in
        # DataLoader workers, or from disk, if a dataset outgrows the runner's memory.
        train_set.cache = True
        validation_set.cache = True

        positive_weights = None
        if is_classification and weighting != "none":
            positive_weights = [
                positive_weight(train_rows[column].to_numpy(), weighting) or 1.0
                for column in columns
            ]

        def build_model() -> Any:
            # Every model gets its own transforms and loss: they are modules the model
            # owns and pickles with itself.
            criterion = None
            if positive_weights is not None:
                from daikonstudio.infrastructure.engines._chemprop_loss import (
                    PositiveWeightedBCELoss,
                )

                criterion = PositiveWeightedBCELoss(positive_weights)
            return _build_model(
                pretrained=pretrained,
                weights_dir=Settings().pretrained_weights_dir,
                hidden=hidden,
                depth=depth,
                is_classification=is_classification,
                output_transform=None
                if target_scaler is None
                else UnscaleTransform.from_standard_scaler(target_scaler),
                n_tasks=len(columns),
                criterion=criterion,
                n_descriptors=0 if x_d_all is None else x_d_all.shape[1],
                x_d_transform=None
                if x_d_scaler is None
                else ScaleTransform.from_standard_scaler(x_d_scaler),
                learning_rate=learning_rate,
            )

        def fit(index: int) -> tuple[Any, Any]:
            """Ensemble member `index` at its selected epoch, and the trainer that fitted
            it: None when an earlier attempt finished it and it was restored instead.

            Members differ only in their seed, which sets the weight initialization and
            the order of the training batches. Data, split and settings are shared."""
            name = (
                _MANIFEST.name
                if members == 1
                else f"{_MANIFEST.name} model {index + 1} of {members}"
            )
            # Only an ensemble saves a finished member: a single model's finished fit is
            # the whole result, which the training run saves itself. A finished member is
            # restored without being rebuilt, so with descriptors on it is fingerprinted
            # by RDKit as well: another release can compute a different descriptor list.
            libraries = ("chemprop", "rdkit") if members > 1 and use_descriptors else ("chemprop",)
            state = training_state_scope(_member_checkpoints(ctx.checkpoints, index), *libraries)
            if members > 1 and state is not None:
                restored = _restored_model(state)
                if restored is not None:
                    ctx.report((index + 1) / members, f"Restored {name} from saved progress")
                    return restored, None

            lightning.seed_everything(ctx.seed + index, workers=True)
            model = build_model()

            def _report_epoch(trainer: Any, _module: Any) -> None:
                # Naming the device is not decoration. `accelerator="auto"` resolves to
                # cuda, mps or cpu depending on what the runner it landed on actually
                # has, nothing validates that a runner registered for the "gpu" lane
                # owns a GPU, and the three do not produce identical numbers. Without
                # this the only honest thing anybody could say about a finished run is
                # "some device".
                ctx.report(
                    (index + (trainer.current_epoch + 1) / epochs) / members,
                    f"Training {name} on {trainer.strategy.root_device}",
                )

            # The validation partition selects the epoch, by `epoch_selection` for an
            # active/inactive target and by `val_loss` (what chemprop's `MPNN` logs, see
            # its `validation_step`) for a measured one. The callback's reasoning, and
            # the sanity-check trap it guards against, live in `_lightning.py`.
            selects_best_epoch = len(validation_set) > 0
            keep_best = (
                keep_best_epoch(by=selection, predict=_validation_scores)
                if is_classification
                else keep_best_epoch()
            )
            callbacks: list[Any] = [
                # Before the reporter, which may raise to stop the fit.
                record_epochs(
                    ctx.record_epoch,
                    epochs=epochs,
                    member=index + 1 if members > 1 else None,
                    members=members if members > 1 else None,
                    unit_scale=float(target_scaler.scale_[0])
                    if target_scaler is not None and len(columns) == 1
                    else None,
                    kept=keep_best if selects_best_epoch else None,
                ),
                LambdaCallback(on_train_epoch_end=_report_epoch),
            ]
            if selects_best_epoch:
                callbacks.append(keep_best)

            # The scratch directory holds the training state a save writes and a resume
            # reads; it only has to outlive `fit`. The trainer does not need it afterwards.
            with tempfile.TemporaryDirectory() as state_dir:
                resume_from = saved_training_state(state, Path(state_dir), model)
                # After the reporter: it raises in `on_train_epoch_end` when the time
                # limit stops the fit, and the save then happens in `on_exception`.
                if state is not None:
                    callbacks.append(save_training_state(state, Path(state_dir)))
                if resume_from is not None:

                    def _report_resume(trainer: Any, _module: Any) -> None:
                        ctx.report(
                            (index + trainer.current_epoch / epochs) / members,
                            f"Resuming {name} from epoch {trainer.current_epoch + 1} of {epochs}",
                        )

                    callbacks.append(LambdaCallback(on_train_start=_report_resume))

                trainer = lightning.Trainer(
                    accelerator="auto",
                    devices=1,
                    max_epochs=epochs,
                    enable_checkpointing=False,
                    logger=False,
                    enable_progress_bar=False,
                    # The interruption point. `report` may raise RunInterrupted, which
                    # propagates out of `fit` and out of `train` -- the only way to stop
                    # work already running on the worker thread.
                    callbacks=callbacks,
                )
                trainer.fit(
                    model,
                    build_dataloader(train_set, batch_size=batch_size, seed=ctx.seed + index),
                    # shuffle=False: `build_dataloader` defaults it to True. Shuffling the
                    # validation loader would not corrupt the loss, but it makes the
                    # number depend on the seed for no reason.
                    build_dataloader(validation_set, batch_size=batch_size, shuffle=False)
                    if len(validation_set) > 0
                    else None,
                    ckpt_path=resume_from,
                    weights_only=False,
                )

            # Restore the selected epoch before anything is scored or saved, so the
            # numbers on the Scorecard and the weights in the artifact are the same
            # model. Restoring in place matters: `trainer.save_checkpoint` below
            # serializes the module the trainer holds, which is this object.
            if keep_best.best_state is not None:
                model.load_state_dict(keep_best.best_state)
            if members == 1:
                return model, trainer
            data = _model_bytes(model)
            # Kept until the run succeeds and its whole checkpoint folder is removed.
            if state is not None and state.save(_FITTED_MODEL, data):
                # Its in-progress state is now hundreds of MB that nothing resumes from.
                state.discard(TRAINING_STATE)
            # A clean copy, as `predict` will load it. The fitted module references its
            # trainer, which holds the optimizer state and the best epoch's copy of the
            # weights; kept, those would stay in memory through every later member.
            return _model_from_bytes(data), None

        fitted = [fit(index) for index in range(members)]
        models = [model for model, _ in fitted]

        # Lightning writes checkpoints to a path, so this round-trips through the
        # filesystem. `tempfile` honours TMPDIR, which is how a deployment points
        # scratch at a fast local NVMe without a setting of our own. The weights
        # serialized here are the restored best epoch's, not the last one's.
        with tempfile.TemporaryDirectory() as scratch:
            checkpoint = Path(scratch) / "model.ckpt"
            stored: dict[str, Any]
            if members == 1:
                fitted[0][1].save_checkpoint(checkpoint)
                # map_location="cpu", as Lightning's own loader does: the weights may
                # sit on a GPU this process cannot read back without it.
                stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
                # The callbacks' state is `keep_best`'s `best_state`, a second full copy
                # of the weights that only a resume reads. Left in, it is stored in every
                # Protocol for nothing: `predict` never looks at it.
                stored.pop("callbacks", None)
            else:
                stored = {_ENSEMBLE: [_model_bytes(model) for model in models]}
            if fill is not None:
                from daikonstudio.infrastructure.chem.featurize import DESCRIPTOR_NAMES

                # Read back by `predict`; Lightning ignores keys it does not know on load.
                stored["daikon_descriptors"] = {
                    "names": list(DESCRIPTOR_NAMES),
                    "fill": [float(v) for v in fill],
                }
            # Protocol 4, not torch's default 2: protocol 2 pickles a `bytes` object as a
            # latin-1 string in UTF-8, which stores an ensemble's models 1.5 times over.
            torch.save(stored, checkpoint, pickle_protocol=4)
            artifact = checkpoint.read_bytes()

        predictor = _predict_trainer()

        def forward(partition: str, rows: pl.DataFrame) -> Any:
            # A prediction-only dataset, rebuilt from the structures rather than
            # reusing `validation_set`: for a regression task that set's targets were
            # scaled in place by `normalize_targets`, while `_forward` returns
            # predictions already unscaled. Scoring the two against each other would
            # compare a real value to a standardized one.
            dataset = MoleculeDataset(
                _datapoints(rows[ctx.structure_column].to_list(), x_d=x_d_for(partition))
            )
            mean, _spread = _ensemble_forward(predictor, models, dataset)
            return mean

        # The validation partition is run once: it picks the cutoffs, then is scored at
        # them. Test is scored at those cutoffs and never sees them chosen.
        validation_values = (
            forward("validation", validation_rows) if validation_rows.height else None
        )
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

        metrics = score(test_rows, forward("test", test_rows))
        validation_metrics = (
            None if validation_values is None else score(validation_rows, validation_values)
        )

        return TrainResult(
            artifact=artifact,
            metrics=metrics,
            validation_metrics=validation_metrics,
            cutoffs=cutoffs or None,
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        _require_chemprop()

        import numpy as np
        import torch
        from chemprop.data import MoleculeDataset
        from chemprop.models import MPNN

        with tempfile.TemporaryDirectory() as scratch:
            checkpoint = Path(scratch) / "model.ckpt"
            checkpoint.write_bytes(ctx.artifact)
            # weights_only=False because the checkpoint pickles its model classes; safe
            # for the reason `_scoring._load_bundle` gives: the artifact is only ever
            # one our own `train` produced. A checkpoint without the key was trained
            # without descriptors and predicts exactly as it always did.
            stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
            extras = stored.get("daikon_descriptors")
            if extras is not None:
                _require_matching_features(
                    {"featurizer": "rdkit_descriptors", "feature_names": extras["names"]}
                )
            # Loaded inside the block, used outside it: the weights are in memory by
            # the time the directory is removed. MPNN.save_hyperparameters() is what
            # makes the architecture recoverable from the checkpoint alone.
            models = (
                [_model_from_bytes(data) for data in stored[_ENSEMBLE]]
                if _ENSEMBLE in stored
                else [MPNN.load_from_checkpoint(checkpoint)]
            )
            # The loaded checkpoint (a one-model one carries its optimizer state) is not
            # needed past this point; held, it would stay in memory through the forward.
            del stored

        structures = ctx.frame[ctx.structure_column].to_list()
        x_d: Any = None
        if extras is not None:
            from daikonstudio.infrastructure.chem.featurize import rdkit_descriptors

            # The training-time preprocessing, with the training-time fill; the model's
            # own X_d_transform does the standardizing.
            logged = _signed_log(rdkit_descriptors(structures))
            x_d = np.where(np.isnan(logged), np.array(extras["fill"]), logged)
        dataset = MoleculeDataset(_datapoints(structures, x_d=x_d))
        values, spread = _ensemble_forward(_predict_trainer(), models, dataset)
        # A checkpoint with fewer tasks would raise an IndexError below, and one with
        # more would drop a task without a word: refuse either with the two counts.
        if values.shape[1] != len(ctx.target_columns):
            raise ValueError(
                f"The model predicts {values.shape[1]} targets, but {len(ctx.target_columns)} "
                "were requested."
            )
        row_ids = pl.Series(range(values.shape[0]), dtype=pl.Int64)

        # Explicit dtypes, matching `_predict_with_tree_ensemble`. An all-None
        # uncertainty list would otherwise infer as polars' Null dtype and make this
        # engine's output schema-incompatible with the ECFP4 engines' for any caller
        # that concatenates or persists results across engines.
        #
        # An ensemble reports its models' spread as the uncertainty. A single model
        # reports none: a fabricated number would be plotted by a triage grid as "the
        # model is confident here", which is worse than an admitted absent one.
        # ponytail: a single model's uncertainty stays null. Upgrade path: an MveFFN head
        # for regression, which predicts a variance without training several models.
        #
        # Long format, one block per target: a joint engine is not wrapped in `FanOut`,
        # so it tags its own rows. A one-target checkpoint yields (n, 1) values and
        # `target_columns` of length one, so it predicts exactly as it always did.
        return pl.concat(
            [
                pl.DataFrame(
                    {
                        "row_id": row_ids,
                        "value": pl.Series(
                            [float(value) for value in values[:, index]], dtype=pl.Float64
                        ),
                        "uncertainty": pl.Series(
                            [float(value) for value in spread[:, index]]
                            if len(models) > 1
                            else [None] * values.shape[0],
                            dtype=pl.Float64,
                        ),
                        "target": pl.Series([column] * values.shape[0], dtype=pl.String),
                    }
                )
                for index, column in enumerate(ctx.target_columns)
            ]
        )
