"""The positive-class-weighted loss chemprop trains with when `positive_weighting` is on.

A module of its own, importing chemprop at the top, because a chemprop checkpoint
pickles its criterion by reference: loading the model must be able to import this class
by its qualified name. Only `chemprop_dmpnn.py` imports this module, and only inside
functions, so the API tier and the default-lane runner -- which have no chemprop -- never
do.
"""

from __future__ import annotations

import torch
from chemprop.nn.metrics import BCELoss
from numpy.typing import ArrayLike
from torch import Tensor
from torch.nn import functional as F


class PositiveWeightedBCELoss(BCELoss):
    """chemprop's BCE with one positive-class weight per task, applied inside the
    logit-space loss exactly as `torch.nn.BCEWithLogitsLoss(pos_weight=...)` does.

    The constructor argument is `positive_weights`, kept on the instance, and the tensor
    lives in the `pos_weight` buffer. chemprop's `MPNN._rebuild_metric` rebuilds a loss
    from its `__dict__` (buffers excluded) by constructor-argument name, so a buffer named
    like the argument would leave it nothing to rebuild from."""

    def __init__(self, positive_weights: list[float], task_weights: ArrayLike = 1.0) -> None:
        super().__init__(task_weights=task_weights)
        self.positive_weights = list(positive_weights)
        weights = torch.tensor(self.positive_weights, dtype=torch.float32)
        self.register_buffer("pos_weight", weights)

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args: object) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            preds, targets, pos_weight=self.pos_weight, reduction="none"
        )
