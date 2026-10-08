import polars as pl
import pytest

from daikonstudio.application.execution.train_protocol import comparison_fractions
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy


def _frame(labels: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"split": labels})


def test_a_computed_split_compares_against_the_fractions_it_asked_for():
    spec = SplitSpec(strategy=SplitStrategy.SCAFFOLD, seed=1, fractions=(0.7, 0.15, 0.15))
    assert comparison_fractions(_frame(["train"] * 7 + ["test"] * 3), spec) == (0.7, 0.15, 0.15)


def test_a_predefined_split_compares_against_the_partitions_it_actually_got():
    """The fractions on a predefined spec are the inert default that __post_init__
    insists means nothing. Holding out 10% against a benchmark that held out 20% is
    not a comparison of split *strategies*, which is the only claim the gap makes."""
    labels = ["train"] * 2651 + ["test"] * 666
    spec = SplitSpec(strategy=SplitStrategy.PREDEFINED, seed=1, column="split")
    train, validation, test = comparison_fractions(_frame(labels), spec)
    assert validation == pytest.approx(0.0)
    assert test == pytest.approx(666 / 3317, abs=1e-9)
    assert train + validation + test == pytest.approx(1.0)
