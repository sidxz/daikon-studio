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
    logit-space loss exactly as `torch.nn.BCEWithLogitsLoss(pos_weight=...)` does."""

    def __init__(self, pos_weight: list[float], task_weights: ArrayLike = 1.0) -> None:
        super().__init__(task_weights=task_weights)
        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args: object) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            preds, targets, pos_weight=self.pos_weight, reduction="none"
        )
