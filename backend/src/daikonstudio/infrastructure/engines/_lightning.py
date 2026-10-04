"""The Lightning pieces the neural engines share.

Its own module for the same reason as `_pretrained.py`: `chemprop-dmpnn` and
`molformer-xl` both train through Lightning and both need the identical
epoch-selection behaviour. The `sanity_checking` guard below is a correctness fix
for a bug that shipped once already, and a guard living in two copies is a guard
that gets fixed in one of them.

No torch or lightning import at module scope. `registry.py` instantiates every
engine at import time, and the API tier and default-lane worker install without
the gpu extra -- so the class is defined inside the factory, where the import is
only paid by a worker that actually trains.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from daikonstudio.application.engines.checkpoints import (
    TRAINING_STATE,
    TRAINING_STATE_SCOPE,
    Checkpoints,
)
from daikonstudio.application.engines.context import RunInterrupted

__all__ = [
    "keep_best_by_validation_loss",
    "save_training_state",
    "saved_training_state",
    "training_state_scope",
]

logger = logging.getLogger(__name__)


def keep_best_by_validation_loss() -> Any:
    """A Lightning callback holding the weights of the lowest-`val_loss` epoch.

    The validation partition selects the epoch. Without this, Lightning computes
    `val_loss` every epoch and nothing reads it: the weights that survive are
    whichever epoch happened to be last, so a model that peaked at epoch 3 and then
    overfit for two more ships in its overfit state and the validation split -- ten
    percent of the dataset -- buys nothing at all.

    Deliberately best-checkpoint selection and NOT early stopping. Early stopping
    needs a patience meaningful relative to `epochs`, and at these engines' defaults
    there is no such value; best-checkpoint uses every epoch the scientist asked for
    and keeps the best one. Early stopping is a compute saving, not a correctness
    fix, and can be added later behind its own condition.

    The caller restores `best_state` before scoring or saving, so the numbers on the
    Scorecard and the weights in the artifact are the same model. `best_state` stays
    `None` when there is no validation dataloader -- nothing is recorded and the
    caller keeps the final epoch.

    Lightning's own `ModelCheckpoint` would do this by writing every candidate to
    disk and reading the winner back; the weights are already in memory and the only
    thing needed is a copy, so this skips the filesystem round-trip and the temporary
    directory that would have to outlive `fit` to make it work.
    """
    from lightning.pytorch.callbacks import Callback

    class _KeepBestByValidationLoss(Callback):
        def __init__(self) -> None:
            self.best_loss = float("inf")
            self.best_state: dict[str, Any] | None = None

        def on_validation_epoch_end(self, trainer: Any, module: Any) -> None:
            # Lightning runs a sanity-check validation pass BEFORE training, and it
            # fires this hook with a perfectly valid `val_loss` measured on the
            # untrained model (verified: called with `sanity_checking=True` at epoch
            # 0, before any optimisation step). Without this guard those random
            # weights are recorded as the best epoch, and any run where no real epoch
            # beats them ships an untrained model with an honest-looking scorecard.
            if trainer.sanity_checking:
                return
            # `val_loss` is the key both engines log. Absent means no validation
            # dataloader, which is legitimate.
            loss = trainer.callback_metrics.get("val_loss")
            if loss is None:
                return
            value = float(loss)
            if value < self.best_loss:
                self.best_loss = value
                # Detached clones: the live tensors keep training after this.
                self.best_state = {
                    key: tensor.detach().clone() for key, tensor in module.state_dict().items()
                }

        # Saved with the training state and restored on resume (Lightning calls these),
        # so best-epoch selection survives a stopped run: without them a resumed fit
        # would forget its best epoch and keep the last one.
        def state_dict(self) -> dict[str, Any]:
            return {"best_loss": self.best_loss, "best_state": self.best_state}

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            self.best_loss = state_dict["best_loss"]
            self.best_state = state_dict["best_state"]

    return _KeepBestByValidationLoss()


def training_state_scope(checkpoints: Checkpoints | None, *libraries: str) -> Checkpoints | None:
    """Where a fit's Lightning training state is saved, fingerprinted with the library
    versions that can read it back: a runner upgraded between attempts starts the fit
    over rather than loading state its torch or lightning cannot."""
    if checkpoints is None:
        return None
    from importlib.metadata import version

    versions = {name: version(name) for name in ("torch", "lightning", *libraries)}
    return checkpoints.scoped(TRAINING_STATE_SCOPE, **versions)


def save_training_state(checkpoints: Checkpoints, scratch: Path) -> Any:
    """A callback saving the trainer's full state at an epoch's end: every
    `checkpoints.interval_seconds`, and once more when the run's time limit stops the
    fit. Verified on Lightning 2.6.5: either save resumes at the next epoch.

    Not on a cancel: the run row is already CANCELLED and the runner API refuses the
    write, so Resume after a cancel continues from the last periodic save.
    """
    import time

    from lightning.pytorch.callbacks import Callback

    class _SaveTrainingState(Callback):
        def __init__(self) -> None:
            self._last = time.monotonic()

        def _save(self, trainer: Any) -> None:
            path = scratch / "training-state.ckpt"
            try:
                trainer.save_checkpoint(path)
                checkpoints.save(TRAINING_STATE, path.read_bytes())
            except Exception:  # best effort: a failed save must not stop the fit
                logger.warning("Could not save training state", exc_info=True)
            self._last = time.monotonic()

        def on_train_epoch_end(self, trainer: Any, module: Any) -> None:
            if time.monotonic() - self._last >= checkpoints.interval_seconds:
                self._save(trainer)

        def on_exception(self, trainer: Any, module: Any, exception: BaseException) -> None:
            if isinstance(exception, RunInterrupted) and not exception.cancelled:
                self._save(trainer)

    return _SaveTrainingState()


def saved_training_state(
    checkpoints: Checkpoints | None, scratch: Path, module: Any
) -> str | None:
    """A path to pass as `ckpt_path`, or None to train from the start.

    Loads the weights into `module` first, strictly, as a check: a state saved by a
    differently built model (shapes changed between attempts) is discarded here
    rather than failing `fit` halfway through restoring. Loading the same weights
    `fit` is about to restore is harmless.
    """
    if checkpoints is None:
        return None
    data = checkpoints.load(TRAINING_STATE)
    if data is None:
        return None
    path = scratch / "resume.ckpt"
    try:
        # Inside the try: a full TMPDIR must train from the start, not fail the run.
        path.write_bytes(data)
        import torch

        # weights_only=False: a Lightning training state pickles chemprop objects (the
        # criterion, the descriptor transform), which weights-only loading refuses. It
        # is the same trust boundary as every saved model this engine loads -- a blob
        # written by this deployment's own runners into the run's workspace. `fit` must
        # be given `weights_only=False` as well: left at its default Lightning hands the
        # path to torch.load, whose default is True, and the restore fails on the
        # checkpoint's hyperparameters.
        state = torch.load(path, map_location="cpu", weights_only=False)
        module.load_state_dict(state["state_dict"])
    except Exception:
        logger.warning(
            "Could not restore the saved training state; training from the start", exc_info=True
        )
        return None
    return str(path)
