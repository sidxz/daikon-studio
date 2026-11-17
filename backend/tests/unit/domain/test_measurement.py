"""The parser's job is to keep a bound a bound. These are the cases that lose data."""

from __future__ import annotations

import polars as pl

from daikonstudio.application.tools import codecs
from daikonstudio.domain.shared.measurement import Relation, parse_measurement
from daikonstudio.infrastructure.tools.measurements import read_measurements


def test_a_bound_keeps_its_direction_and_never_reads_as_exact() -> None:
    for raw, relation in (
        (">64", Relation.GREATER),
        ("<0.01", Relation.LESS),
        (">=10", Relation.AT_LEAST),
        ("<=10", Relation.AT_MOST),
        ("=>10", Relation.AT_LEAST),
        ("≥10", Relation.AT_LEAST),
        ("≤10", Relation.AT_MOST),
    ):
        reading = parse_measurement(raw)
        assert reading is not None, raw
        assert reading.relation is relation, raw
        assert reading.is_censored, raw


def test_a_plain_number_is_exact_and_a_non_reading_is_none() -> None:
    exact = parse_measurement("12.5")
    assert exact is not None and exact.relation is Relation.EXACT and exact.value == 12.5
    assert parse_measurement(7) is not None
    assert parse_measurement("1,250") == parse_measurement("1250")
    # A tilde is a point estimate, not a bound: it must not be reported as censored.
    approximate = parse_measurement("~5")
    assert approximate is not None and not approximate.is_censored
    for raw in (None, "", "NT", "not tested", float("nan"), float("inf"), True):
        assert parse_measurement(raw) is None, raw


def test_a_column_splits_into_values_relations_and_two_separate_counts() -> None:
    readings = read_measurements(pl.Series("IC50", ["12.5", ">64", "NT", None, "<0.01"]))
    assert readings.values.to_list() == [12.5, 64.0, None, None, 0.01]
    assert readings.relations.to_list() == ["=", ">", None, None, "<"]
    assert readings.relations.name == "IC50_relation"
    # The bound's number survives in `values`, and is withheld from anything averaging.
    assert readings.exact_only.to_list() == [12.5, None, None, None, None]
    assert (readings.n_censored, readings.n_unreadable) == (2, 1)


def test_a_relation_column_is_shown_to_the_agent_and_not_mistaken_for_structures() -> None:
    """`=` is a SMILES character and a SMILES mark, so the content test used to withhold
    the one column that says which readings are bounds."""
    frame = pl.DataFrame(
        {"id": ["a", "b", "c"], "IC50": [1.0, 64.0, 2.0], "IC50_relation": ["=", ">", "="]}
    )
    _, summary = codecs.encode("property_table", frame)
    assert "IC50_relation" not in summary["withheld"]
