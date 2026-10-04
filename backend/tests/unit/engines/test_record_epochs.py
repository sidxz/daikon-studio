"""The one recorder both neural engines use: what an epoch's logged numbers become."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

pytest.importorskip("lightning")

from daikonstudio.application.engines.context import EpochPoint
from daikonstudio.infrastructure.engines._lightning import record_epochs


def _end_of_epoch(logged: dict[str, float], **kwargs: object) -> EpochPoint:
    points: list[EpochPoint] = []
    trainer = SimpleNamespace(
        callback_metrics=logged,
        current_epoch=4,
        strategy=SimpleNamespace(root_device="cuda:0"),
    )
    record_epochs(points.append, epochs=30, **kwargs).on_train_epoch_end(trainer, None)
    return points[0]


def test_an_epoch_reads_the_epoch_mean_loss_and_names_scores_in_the_shared_vocabulary():
    point = _end_of_epoch(
        {
            "train_loss": 0.9,  # the last batch
            "train_loss_epoch": 0.6,
            "val_loss": 0.5,
            "val/roc": 0.81,
            "val/prc": 0.42,
            "val/binary-mcc": 0.3,
        },
        member=2,
        members=5,
    )

    assert (point.epoch, point.epochs) == (5, 30)  # 1-based
    assert (point.train_loss, point.val_loss) == (0.6, 0.5)
    assert point.scores == {"auroc": 0.81, "auprc": 0.42, "mcc": 0.3}
    assert (point.member, point.members, point.device) == (2, 5, "cuda:0")


def test_regression_errors_return_to_the_targets_own_unit():
    point = _end_of_epoch(
        {"val_loss": 1.1, "val/rmse": 1.1, "val/mae": 0.9, "val/r2": 0.2}, unit_scale=10.0
    )

    assert point.scores == pytest.approx({"rmse": 11.0, "mae": 9.0, "r2": 0.2})


def test_errors_with_no_unit_to_return_to_are_left_out_rather_than_mislabeled():
    """A fit over several measured targets at once pools them, so there is no one unit."""
    point = _end_of_epoch({"val/rmse": 1.1, "val/mae": 0.9, "val/r2": 0.2})

    assert point.scores == {"r2": 0.2}


def test_a_score_that_is_not_a_number_is_left_out():
    point = _end_of_epoch({"val_loss": math.nan, "val/roc": math.nan, "val/prc": 0.4})

    assert point.val_loss is None
    assert point.scores == {"auprc": 0.4}
