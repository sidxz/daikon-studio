"""The two ends of sparseness: nothing to learn from, and nothing to score on.

They get opposite treatments on purpose. No labelled training row is the
mistyped-column case and the dataset is refused, because every model of that target
would be vacuous. No labelled test row is a real possibility for a rare assay under a
scaffold split, and the run continues with that target's metrics undefined -- the same
degradation every other unscoreable metric already gets.
"""

from __future__ import annotations

import polars as pl

from daikonstudio.application.data.create_dataset import _degenerate_partition
from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.domain.data.target import TargetKind, TargetSpec
from daikonstudio.domain.shared.errors import ValidationError

TOX = TargetSpec(column="tox", kind=TargetKind.BINARY)


def test_a_target_with_no_labelled_training_rows_is_refused() -> None:
    frame = pl.DataFrame(
        {
            "tox": [None, None, 1.0, 0.0],
            "split": ["train", "train", "test", "test"],
        }
    )

    error = _degenerate_partition(frame, TOX)

    assert isinstance(error, ValidationError)
    assert "tox" in str(error)
    # "train set", matching the single-class message in the same function rather than
    # introducing a second name for the same partition two lines apart.
    assert "no measured compounds in the train set" in str(error)


def test_a_target_measured_in_training_is_allowed() -> None:
    frame = pl.DataFrame(
        {
            "tox": [1.0, 0.0, None, None],
            "split": ["train", "train", "test", "test"],
        }
    )

    assert _degenerate_partition(frame, TOX) is None


def test_a_target_with_no_labelled_test_rows_is_undefined_not_empty() -> None:
    """A rare assay under a scaffold split can land every one of its measurements in
    training. The run continues -- the other targets are fine -- and this one reports
    zero measured rows rather than a card full of fabricated zeros or a crash on an
    empty array."""
    from daikonstudio.application.execution.build_scorecard import (
        build_scorecard,
        held_out_chemistry,
    )
    from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer

    card = build_scorecard(
        target="tox",
        task=TaskType.BINARY_CLASSIFICATION,
        metrics={"mcc": None},
        engine_id="ecfp4-randomforest",
        conditions={},
        baseline_engine_id="ecfp4-randomforest",
        baseline_metrics=None,
        baseline_is_self=False,
        actual=[],
        predicted=[],
        structures=[],
        chemistry=held_out_chemistry([], ["CCO"], RdkitStructureNormalizer()),
        target_unit=None,
        target_direction=None,
        split_strategy="scaffold",
    )

    assert card.labelled_test_rows == 0
    assert card.primary_metric_ci is None
    assert card.metrics["mcc"] is None


def test_an_unmeasured_column_still_reports_every_metric_as_a_null_entry() -> None:
    """What keeps each split draw's per-metric lists aligned with `replicate_seeds`.

    The lists are read positionally: draw 2 contributing no entry for a metric would
    shift draw 3's score into draw 2's slot and report one draw's spread as another's.
    That cannot happen while a column with no labelled rows in a draw's test partition
    still produces every metric key with a null value -- which is the behaviour pinned
    here, because it is what makes a defensive hole-filler unnecessary rather than an
    accident that happens to hold today.
    """
    import numpy as np

    from daikonstudio.application.execution.train_protocol import _measured
    from daikonstudio.infrastructure.engines._scoring import (
        classification_metrics,
        labelled_only,
    )

    unmeasured = np.array([float("nan"), float("nan")])
    probabilities = np.array([0.7, 0.2])
    labels, predicted, scores = labelled_only(
        unmeasured, (probabilities >= 0.5).astype(float), probabilities
    )
    assert len(labels) == 0

    metrics, undefined = _measured(
        classification_metrics(labels, predicted, scores, train_has_both_classes=True)
    )

    assert set(metrics) == {"mcc", "balanced_accuracy", "auroc", "auprc"}
    assert all(value is None for value in metrics.values())
    assert undefined == set(metrics)
