"""The polars side of `domain.shared.measurement`: one mixed column, two typed ones.

A tool reads an activity column that holds `12.5`, `>64` and `NT` together. The cast
every caller used to write (`cast(Float64, strict=False)`) turns the second and third
into the same null, so a bound and a blank become indistinguishable and the compounds
that failed the assay leave without a trace. This splits the column instead: the
numbers in one series, the relations in another, and a count of each kind so a tool
can say what it left out.

Lives here rather than in `impl/` because more than one tool reads activity, and the
cost of a second parser is two tools that disagree about what `>64` means.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from daikonstudio.domain.shared.measurement import (
    RELATION_SUFFIX,
    Relation,
    parse_measurement,
)


@dataclass(frozen=True, kw_only=True, slots=True)
class Readings:
    """A parsed activity column. `values` and `relations` align with the input rows."""

    values: pl.Series
    relations: pl.Series
    # Rows that parsed to a bound, and rows that were not readings at all ('NT', ''). Kept
    # apart because "left out 9 censored" and "left out 9 unreadable" ask for different
    # things from a scientist.
    n_censored: int
    n_unreadable: int

    @property
    def exact_only(self) -> pl.Series:
        """The values with every bound nulled: what a correlation or a mean may use."""
        return pl.Series(
            name=self.values.name,
            values=[
                None if relation != Relation.EXACT else value
                for value, relation in zip(self.values, self.relations, strict=True)
            ],
            dtype=pl.Float64,
        )


def read_measurements(column: pl.Series) -> Readings:
    """Split a column of readings into aligned value and relation series."""
    values: list[float | None] = []
    relations: list[str | None] = []
    n_censored = n_unreadable = 0
    for raw in column.to_list():
        reading = parse_measurement(raw)
        if reading is None:
            values.append(None)
            relations.append(None)
            # A blank cell was never a reading anybody lost; only a non-empty one counts.
            if raw is not None and str(raw).strip() != "":
                n_unreadable += 1
            continue
        values.append(reading.value)
        relations.append(str(reading.relation))
        if reading.is_censored:
            n_censored += 1
    return Readings(
        values=pl.Series(name=column.name, values=values, dtype=pl.Float64),
        relations=pl.Series(name=column.name + RELATION_SUFFIX, values=relations, dtype=pl.String),
        n_censored=n_censored,
        n_unreadable=n_unreadable,
    )
