"""Which epoch a neural fit keeps (`epoch_selection`), and the averaged scores it uses."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("lightning")

import torch

from daikonstudio.infrastructure.engines._lightning import (
    averaged_scores,
    keep_best_epoch,
    record_epochs,
)

NAN = math.nan


class _Module:
    def __init__(self) -> None:
        self.weight = torch.tensor([0.0])
        self.logged: dict[str, float] = {}

    def state_dict(self) -> dict[str, Any]:
        return {"weight": self.weight}

    def log(self, key: str, value: float) -> None:
        self.logged[key] = value


def _epoch(callback: Any, module: _Module, epoch: int, loss: float, scores: list[float]) -> None:
    module.weight = torch.tensor([float(epoch)])
    targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    trainer = SimpleNamespace(
        sanity_checking=False,
        current_epoch=epoch - 1,
        callback_metrics={"val_loss": torch.tensor(loss)},
    )
    callback.on_validation_epoch_start(trainer, module)
    callback.on_validation_batch_end(
        trainer, module, None, (torch.tensor(scores).reshape(-1, 1), targets), 0
    )
    callback.on_validation_epoch_end(trainer, module)


def _fit(by: str) -> tuple[Any, _Module]:
    """Epoch 1 has the lower loss and ranks one active below an inactive; epoch 2's
    loss is higher, from overconfidence, but it ranks every active first."""
    callback = keep_best_epoch(by=by, predict=lambda _module, batch: batch)
    module = _Module()
    _epoch(callback, module, 1, loss=0.4, scores=[0.7, 0.6, 0.55, 0.3])
    _epoch(callback, module, 2, loss=0.9, scores=[0.99, 0.2, 0.9, 0.01])
    return callback, module


def test_a_classification_fit_keeps_the_best_ranking_epoch_not_the_lowest_loss():
    callback, module = _fit("auprc")

    assert (callback.best_epoch, callback.kept_by) == (2, "auprc")
    assert torch.equal(callback.best_state["weight"], torch.tensor([2.0]))
    assert module.logged == {"val/roc-mean": 1.0, "val/prc-mean": 1.0}


def test_lowest_loss_remains_a_choice():
    callback, _module = _fit("loss")

    assert callback.best_epoch == 1


def test_without_a_score_to_compare_the_loss_decides():
    """A validation set with no active for any target has no PR AUC."""
    callback = keep_best_epoch(by="auprc", predict=lambda _module, batch: batch)
    module = _Module()
    targets = torch.zeros(3, 1)
    for epoch, loss in ((1, 0.3), (2, 0.5)):
        trainer = SimpleNamespace(
            sanity_checking=False,
            current_epoch=epoch - 1,
            callback_metrics={"val_loss": torch.tensor(loss)},
        )
        callback.on_validation_epoch_start(trainer, module)
        callback.on_validation_batch_end(trainer, module, None, (torch.rand(3, 1), targets), 0)
        callback.on_validation_epoch_end(trainer, module)

    assert (callback.best_epoch, callback.kept_by) == (1, "loss")
    assert module.logged == {}


def test_each_target_counts_once_however_many_compounds_it_labels():
    # Target a: perfectly ranked, 6 labelled. Target b: inverted, 2 labelled, the rest
    # unlabelled (NaN). Target c: one class only, so it has no score and is left out.
    scores = torch.tensor(
        [
            [0.9, 0.1, 0.5],
            [0.8, 0.9, 0.5],
            [0.7, 0.5, 0.5],
            [0.3, 0.5, 0.5],
            [0.2, 0.5, 0.5],
            [0.1, 0.5, 0.5],
        ]
    )
    targets = torch.tensor(
        [[1, 1, 0], [1, 0, 0], [1, NAN, 0], [0, NAN, 0], [0, NAN, 0], [0, NAN, 0]]
    )

    averaged = averaged_scores(scores, targets)

    assert averaged["auroc"] == pytest.approx((1.0 + 0.0) / 2)
    assert averaged["auprc"] == pytest.approx((1.0 + 0.5) / 2)


def test_the_run_page_gets_the_averaged_scores_and_the_kept_epoch():
    points: list[Any] = []
    trainer = SimpleNamespace(
        callback_metrics={"val_loss": 0.9, "val/prc": 0.4, "val/prc-mean": 0.6, "val/roc": 0.8},
        current_epoch=6,
        strategy=SimpleNamespace(root_device="cpu"),
    )
    kept = SimpleNamespace(best_epoch=5, kept_by="auprc")

    record_epochs(points.append, epochs=30, kept=kept).on_train_epoch_end(trainer, None)

    assert points[0].scores == {"auprc": 0.6, "auroc": 0.8}
    assert (points[0].kept_epoch, points[0].kept_by) == (5, "auprc")
