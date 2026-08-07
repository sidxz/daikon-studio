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

from typing import Any

__all__ = ["keep_best_by_validation_loss"]


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

    return _KeepBestByValidationLoss()
