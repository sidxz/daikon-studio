"""What the table profile must never get wrong.

Two things, and the second matters more. It must stay ignorant of what the table
is about, because a profiler that hunts for activity columns and censored values
is one use case written into the plumbing and is wrong the moment the data is
something else. And it must not let a column whose values are the asset leak
them, including through the values that failed to parse.
"""

from __future__ import annotations

import pytest

from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.tables.profiler import PolarsTableProfiler

_CSV = (
    b"id,structure,IC50_uM,plate,run_date\n"
    b"A-1,CCO,1.2,P1,2026-03-04\n"
    b"A-2,c1ccccc1,>50,P1,2026-03-04\n"
    b"A-3,CCN,0.4,P2,2026-06-18\n"
    b"A-4,CCO,>50,P2,2026-06-18\n"
)


def _column(csv: bytes, name: str):
    profile = PolarsTableProfiler().profile(csv, filename="t.csv")
    return next(c for c in profile.columns if c.name == name)


def test_it_reports_what_parsers_made_of_a_column_rather_than_what_it_means() -> None:
    structure = _column(_CSV, "structure")
    assert structure.parses_as["molecule"] == 4
    # No "is_structure_column" flag anywhere: the model infers that, and a rule
    # here would be this one use case hardcoded.
    assert not hasattr(structure, "is_structure")


def test_values_that_do_not_fit_are_shown_without_knowing_what_they_are() -> None:
    """Nothing here knows what censoring is. It reports that a mostly-numeric
    column has entries that are not numbers, and shows them; a model reads
    '>50' as a lower bound."""
    activity = _column(_CSV, "IC50_uM")
    assert activity.parses_as["number"] == 2
    assert ">50" in activity.odd_values
    # De-duplicated: two rows reading ">50" is one fact.
    assert activity.odd_values.count(">50") == 1


def test_a_structure_column_withholds_its_values() -> None:
    structure = _column(_CSV, "structure")
    assert structure.withheld is True
    assert structure.examples == ()
    assert structure.most_repeated is not None


def test_a_withheld_column_does_not_leak_through_its_failures() -> None:
    """A structure that failed to parse is still the scientist's compound.
    Quoting the failures of a column whose values are deliberately not shown
    would undo the withholding entirely."""
    csv = _CSV + b"A-5,not-a-molecule-at-all,0.9,P2,2026-06-18\n"
    structure = _column(csv, "structure")
    assert structure.withheld is True
    assert structure.odd_values == ()
    # The count still travels, because the scientist needs to know.
    assert structure.odd_count == 1


def test_repeats_are_reported_without_calling_them_replicates() -> None:
    structure = _column(_CSV, "structure")
    assert structure.distinct == 3
    assert structure.filled == 4
    assert structure.most_repeated == ("CCO", 2) or structure.most_repeated[1] == 2


def test_a_short_categorical_is_not_mistaken_for_a_molecule() -> None:
    """A bare element symbol parses as a molecule and is also a perfectly
    ordinary label, so one-character values are not evidence of structures."""
    csv = b"id,grade\n1,N\n2,C\n3,N\n"
    grade = _column(csv, "grade")
    assert grade.parses_as.get("molecule", 0) == 0
    assert grade.withheld is False


def test_something_that_is_not_a_table_is_refused() -> None:
    with pytest.raises(ValidationError):
        PolarsTableProfiler().profile(b"", filename="empty.csv")
