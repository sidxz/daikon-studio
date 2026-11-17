"""A description of a table's shape, with no idea what the table is about.

The planner works far better looking at the data than guessing about it. The
first three questions it asked on a real goal -- which column is the readout, are
there censored values, are any compounds measured twice -- are all answerable by
reading the file, and a question that could have been answered by reading the
first ten rows is an irritation rather than a decision.

**The profile must not know what it is looking at.** A profiler that hunts for an
activity column, counts censored values and clusters scaffolds is this one use
case written into the plumbing, and it is wrong the moment the table holds
protein variants or a screening deck. It is also the way such a thing goes
quietly wrong: the sibling SAR project's first activity-column picker matched
"Kinase" on "ki" and "Max Phase" on "ph".

So the profile reports only facts about a table, and the inference is the model's:

- Do not look for censored values. Report that a mostly-numeric column has three
  entries that are not numbers, and show them. A model reads ">50" as censoring.
- Do not look for an activity column. Report which columns are numeric and what
  range they cover.
- Do not look for replicates. Report which columns repeat values and how often.
- Do not look for structures. Report, for every parser this server owns, how many
  of a column's values it accepts.

That last one is the extensible bit: the profile's domain awareness is a list of
parsers, not a list of rules. Teaching it a new kind of data is a parser, and
every column gets described against it for free.

Anything needing real domain knowledge -- scaffold diversity, assay semantics --
is a tool the plan runs, not a field here. That line is what keeps this generic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, kw_only=True)
class ColumnProfile:
    name: str
    filled: int
    distinct: int
    # How many values each parser accepted, keyed by parser name. A column where
    # "molecule" accepts nearly everything is a structure column, and the model
    # can say so without this code having an opinion.
    parses_as: dict[str, int]
    # A few values, verbatim. Omitted for a column whose values are the asset --
    # see `withheld`.
    examples: tuple[str, ...] = ()
    # True when the examples were deliberately not included because the column
    # holds the thing worth protecting. The shape still goes to the model; the
    # structures do not.
    withheld: bool = False
    # Values that failed the parser the rest of the column passes. Where
    # censoring, placeholders, units and typos all surface, without this code
    # needing a concept for any of them.
    odd_values: tuple[str, ...] = ()
    # How many values failed, whether or not any are quoted above. A withheld
    # column reports the count and nothing else: a structure that did not parse
    # is still a compound, and quoting it would undo the withholding.
    odd_count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    median: float | None = None
    # The most repeated value and how many times, when anything repeats. Enough
    # for a model to spot a batch marker or a replicated identifier.
    most_repeated: tuple[str, int] | None = None


@dataclass(frozen=True, kw_only=True)
class TableProfile:
    rows: int
    # True when only the first `rows` were read. A profile that silently
    # described a sample as though it were the whole table would make every
    # count in it a lie.
    truncated: bool = False
    columns: tuple[ColumnProfile, ...] = ()


class TableProfiler(Protocol):
    def profile(self, content: bytes, *, filename: str) -> TableProfile:
        """Describe the table in `content`.

        Raises `ValidationError` when the bytes are not a table this server can
        read. Takes bytes rather than a path because an implementation may run in
        another process, and only plain data crosses that boundary.
        """
        ...
