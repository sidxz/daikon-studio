"""Why a metric has no value, in words that fit the task it came from.

Before Spearman joined the regression vocabulary, an undefined metric was
always a classification metric, so the reason could talk about classes. A rank
correlation is undefined whenever one side has no order to correlate, which a
regression target reaches on its own, and the advice then has to stop naming a
class that does not exist.
"""

from __future__ import annotations

import polars as pl

from daikonstudio.application.execution.train_protocol import _undefined_reasons


def rows(values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"potency": values})


def test_a_single_valued_regression_test_set_is_explained_without_classes() -> None:
    reason = _undefined_reasons(
        {"spearman"},
        "potency",
        train_rows=rows([1.0, 2.0, 3.0, 4.0]),
        test_rows=rows([5.0, 5.0, 5.0]),
    )

    assert reason is not None
    assert "class" not in reason["spearman"]
    assert "potency" in reason["spearman"]


def test_a_single_valued_training_set_is_explained_without_classes() -> None:
    reason = _undefined_reasons(
        {"spearman"},
        "potency",
        train_rows=rows([7.0, 7.0, 7.0]),
        test_rows=rows([1.0, 2.0, 3.0]),
    )

    assert reason is not None
    assert "class" not in reason["spearman"]


def test_a_cause_the_split_cannot_explain_is_not_attributed_to_the_split() -> None:
    """Both sides vary, so the metric is undefined for some other reason --
    constant predictions, most likely. Saying so beats naming a cause that was
    just ruled out."""
    reason = _undefined_reasons(
        {"spearman"},
        "potency",
        train_rows=rows([1.0, 2.0, 3.0]),
        test_rows=rows([4.0, 5.0, 6.0]),
    )

    assert reason == {"spearman": "Undefined: the engine returned no value for this metric."}


def test_nothing_undefined_means_no_reasons() -> None:
    assert _undefined_reasons(set(), "potency", rows([1.0]), rows([2.0])) is None
