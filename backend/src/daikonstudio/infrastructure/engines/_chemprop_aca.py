"""Cliff awareness: a triplet term on chemprop's latent vectors, added to its loss.

Implemented from the equations in Cui et al., "Activity-cliff awareness enables robust
graph learning for molecular property prediction", Nature Communications (2026),
doi:10.1038/s41467-026-75713-2. Not from the authors' reference implementation
(github.com/shenwanxiang/ACANet, the `clsar` package), which is GPL-3.0 and so could not
be taken into this codebase; only the paper's stated arithmetic is reproduced here.

A regression loss on its own rewards a latent space in which structurally similar
molecules sit close together and so receive similar predictions. An activity cliff is
precisely the case where that assumption is false, and a model trained without this term
splits the difference between the two compounds and gets both wrong. The term mines each
training batch for triplets -- an anchor, a compound of near-equal activity, and one of
very different activity -- and penalises the triplets whose arrangement in the latent
space contradicts the arrangement of their labels. Triplets that are already arranged
correctly score zero, so the count mined falls as training progresses.

A module of its own, importing chemprop at the top, because `chemprop_dmpnn.py` imports
it inside a function and neither the API tier nor the default-lane runner installs
chemprop. Nothing here is needed to predict: the term is a training-time gradient and
leaves no trace in the fitted weights, which is why `predict` loads every artifact as a
plain `MPNN` and never imports this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from chemprop.models import MPNN
from torch import Tensor

# Anchors held in flight at once. The mining compares every (anchor, positive, negative)
# combination, so a whole batch at once would allocate batch^3 floats -- 537 MB at the
# largest batch size this engine offers. Chunking the anchors bounds that at 32 * batch^2
# (33 MB) and changes no result.
# ponytail: a fixed 32. Tie it to the batch size if a runner with little memory trips on
# it, or if profiling shows the loop costs more than the allocation saves.
_ANCHOR_CHUNK = 32


@dataclass(frozen=True)
class CliffAwareness:
    """What counts as a cliff, and how much it counts.

    `lower` and `upper` are activity differences in the units the loss actually sees --
    standardized ones, since a regression fit scales its targets -- so the engine divides
    the scientist's figures, which are in the target's own units, by the scaler's scale.
    """

    lower: float
    upper: float
    weight: float


def cliff_triplet_loss(
    latent: Tensor, labels: Tensor, measured: Tensor, *, lower: float, upper: float
) -> Tensor:
    """The batch's cliff-awareness term: a squared soft-margin triplet loss.

    `latent` is (rows, features); `labels` and `measured` are one target's standardized
    values and whether each row was measured, both of length rows. Unmeasured rows take
    no part -- this engine is handed the sparse frame whole, so a row may hold no value
    for the target being fitted.

    A triplet is an anchor a, a positive p whose activity differs from it by at most
    `lower`, and a negative n whose activity differs by at least `upper`. It scores

        relu( d(a,p) - d(a,n) + |y_a - y_n| - |y_a - y_p| ) ** 2

    with d the Euclidean distance between latent vectors. The margin is the one the
    labels themselves imply rather than a constant, so a triplet scores zero as soon as
    the latent space separates the three by as much as their activities do, and
    contributes no gradient from then on. The mean is taken over the triplets that score
    above zero -- the paper's "high-value" ones -- so a batch of mostly settled triplets
    does not dilute the few that are still wrong.

    Squared because chemprop's regression head is trained on mean squared error, and the
    two terms are summed: the paper's unsquared form is the one that pairs with a mean
    absolute error head.
    """
    rows = latent[measured]
    values = labels[measured]
    # Two measured rows cannot make an anchor, a positive and a negative between them.
    if rows.shape[0] < 3:
        return latent.new_zeros(())

    label_gap = (values[:, None] - values[None, :]).abs()
    latent_gap = torch.cdist(rows, rows)
    is_positive = label_gap <= lower
    # A compound is not its own positive: it would pair at distance zero and drag the
    # mean towards a margin no real pair has to meet.
    is_positive.fill_diagonal_(False)
    is_negative = label_gap >= upper

    total = latent.new_zeros(())
    mined = 0
    for start in range(0, rows.shape[0], _ANCHOR_CHUNK):
        stop = start + _ANCHOR_CHUNK
        # (anchors, positives, negatives) throughout: positives vary down the second
        # axis, negatives across the third.
        scored = torch.relu(
            latent_gap[start:stop, :, None]
            - latent_gap[start:stop, None, :]
            + label_gap[start:stop, None, :]
            - label_gap[start:stop, :, None]
        ).square()
        scored = scored * (is_positive[start:stop, :, None] & is_negative[start:stop, None, :])
        total = total + scored.sum()
        mined += int((scored > 0).sum())
    # No triplet mined is not an error: a batch may hold no cliff at all, and early in
    # training on an easy dataset most batches do not.
    return total / max(mined, 1)


class CliffAwareMPNN(MPNN):
    """chemprop's network with the cliff-awareness term added to its training loss.

    `cliff` is attached after construction rather than declared as a constructor
    argument. `MPNN.__init__` saves its own arguments as the checkpoint's
    hyperparameters, and a subclass that added one would write a checkpoint that
    `MPNN.load_from_file` -- how `predict` loads every artifact this engine saves --
    could not rebuild.

    `training_step` repeats chemprop's own body instead of calling `super()` and adding
    to what it returns: the term needs the latent vectors, and reaching them through a
    second `fingerprint` call would run the message passing over every batch twice.
    """

    cliff: CliffAwareness | None = None

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        latent = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor.train_step(latent)
        loss: Tensor = self.criterion(preds, targets, mask, weights, lt_mask, gt_mask)
        self.log(
            "train_loss",
            self.criterion,
            batch_size=self.get_batch_size(batch),
            prog_bar=True,
            on_epoch=True,
        )

        if self.cliff is None:
            return loss
        # The vector the predictor reads, which with RDKit descriptors on includes them:
        # the geometry worth reshaping is the one the head actually sees. Column 0 is the
        # only column -- the engine refuses cliff awareness on a multi-target fit.
        return loss + self.cliff.weight * cliff_triplet_loss(
            latent,
            targets[:, 0],
            mask[:, 0],
            lower=self.cliff.lower,
            upper=self.cliff.upper,
        )
