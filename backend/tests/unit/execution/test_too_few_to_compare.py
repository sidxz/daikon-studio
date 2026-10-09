"""A re-split's scores are withheld when its test rows hold too few of a class.

The case this exists for: the Lane 2018 Mtb set at a 3.4% active rate. The optimism-gap
leg keeps the Dataset's partition sizes, so a 151-row test set drew about five actives
at the 100 nM cutoff, every classification metric came back at exactly 1.000, and the
Scorecard printed an optimism gap of 0.497 from it. The 1 uM and 10 uM targets, from the
same fit on the same rows, gave sensible gaps -- which is what makes it a threshold
problem rather than a broken leg.
"""

from __future__ import annotations

import polars as pl

from daikonstudio.application.engines.context import MIN_CUTOFF_CLASS_COUNT
from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.application.execution.train_protocol import _too_few_to_compare


def labels(positives: int, negatives: int) -> pl.DataFrame:
    return pl.DataFrame({"active": [1.0] * positives + [0.0] * negatives})


def test_a_handful_of_actives_is_not_compared() -> None:
    reason = _too_few_to_compare("active", TaskType.BINARY_CLASSIFICATION, labels(5, 146))

    assert reason is not None
    assert "5 active" in reason and "146 inactive" in reason


def test_enough_of_both_classes_is_compared() -> None:
    assert (
        _too_few_to_compare(
            "active",
            TaskType.BINARY_CLASSIFICATION,
            labels(MIN_CUTOFF_CLASS_COUNT, MIN_CUTOFF_CLASS_COUNT),
        )
        is None
    )


def test_the_rare_class_can_be_either_one() -> None:
    """An overwhelmingly active test set is as uninformative as an overwhelmingly
    inactive one; the metric does not care which side is thin."""
    assert (
        _too_few_to_compare("active", TaskType.BINARY_CLASSIFICATION, labels(146, 5)) is not None
    )


def test_regression_has_no_class_to_count() -> None:
    """Guarding a continuous target on a class count would withhold every re-split
    score a regression Protocol has."""
    frame = pl.DataFrame({"potency": [1.0, 2.0, 3.0]})

    assert _too_few_to_compare("potency", TaskType.REGRESSION, frame) is None
