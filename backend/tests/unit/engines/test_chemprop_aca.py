"""The arithmetic of the cliff-awareness term, on three compounds worked by hand.

Skipped unless chemprop imports, because the module under test imports it: the API tier
and the default-lane worker deliberately install without it.

Three compounds, one measured target, `lower=0.2` and `upper=1.0`. Activities 0.0, 0.1
and 3.0 put exactly two triplets in range -- (anchor 0, positive 1, negative 2) and
(anchor 1, positive 0, negative 2) -- because compound 2 has no compound of near-equal
activity to serve as its positive. Latent vectors are one-dimensional so that the
distance between two of them is the difference between two numbers, which is what makes
the expected loss something a reader can check rather than take on trust.
"""

from __future__ import annotations

import pytest

pytest.importorskip("chemprop")

import torch

from daikonstudio.infrastructure.engines._chemprop_aca import cliff_triplet_loss

_LABELS = torch.tensor([0.0, 0.1, 3.0])
_MEASURED = torch.tensor([True, True, True])
_RANGE = {"lower": 0.2, "upper": 1.0}


def test_a_latent_space_that_already_matches_the_labels_costs_nothing() -> None:
    """The premise of the term. The two compounds of near-equal activity sit on top of
    each other and the third is five away, so both triplets are already arranged as their
    activities say they should be: the term must add nothing to the loss and send back no
    gradient, or it would be a penalty on every batch rather than on the cliffs."""
    latent = torch.tensor([[0.0], [0.0], [5.0]], requires_grad=True)

    loss = cliff_triplet_loss(latent, _LABELS, _MEASURED, **_RANGE)

    assert loss.item() == 0.0
    loss.backward()
    assert latent.grad is not None
    assert torch.count_nonzero(latent.grad).item() == 0


def test_a_cliff_the_network_has_collapsed_costs_what_the_paper_says_it_should() -> None:
    """The arrangement the term exists to correct: the two compounds whose activities
    differ by 3.0 have been placed on top of each other, and the pair of near-equal
    activity five apart.

    Worked by hand, with the margin each triplet's own labels imply:
      (0, 1, 2): relu(5 - 0 + |0.0-3.0| - |0.0-0.1|) ** 2 = 7.9 ** 2 = 62.41
      (1, 0, 2): relu(5 - 5 + |0.1-3.0| - |0.1-0.0|) ** 2 = 2.8 ** 2 =  7.84
    and the mean over the two that scored above zero is 35.125. An axis or a sign the
    wrong way round lands somewhere else.
    """
    latent = torch.tensor([[0.0], [5.0], [0.0]])

    loss = cliff_triplet_loss(latent, _LABELS, _MEASURED, **_RANGE)

    assert loss.item() == pytest.approx(35.125)


def test_an_unmeasured_compound_takes_no_part() -> None:
    """This engine is handed the sparse frame whole, so a row may hold no value for the
    target being fitted. A fourth compound at 3.05 would be a positive for compound 2 and
    open triplets that do not exist; unmeasured, it must leave the loss exactly as the
    three measured compounds leave it."""
    latent = torch.tensor([[0.0], [5.0], [0.0], [0.0]])
    labels = torch.tensor([0.0, 0.1, 3.0, 3.05])
    measured = torch.tensor([True, True, True, False])

    loss = cliff_triplet_loss(latent, labels, measured, **_RANGE)

    assert loss.item() == pytest.approx(35.125)


def test_a_batch_with_no_cliff_in_it_is_not_an_error() -> None:
    """Early in training, and on an easy dataset throughout, most batches hold no pair
    far enough apart to mine. That is a quiet zero, not a division by zero."""
    labels = torch.tensor([0.0, 0.05, 0.1])

    loss = cliff_triplet_loss(torch.tensor([[0.0], [5.0], [0.0]]), labels, _MEASURED, **_RANGE)

    assert loss.item() == 0.0
