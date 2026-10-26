"""The study map: what it refuses, and what it says about what it drew."""

from __future__ import annotations

import polars as pl
import pytest

from daikonstudio.infrastructure.tools.impl.library import MAP_FLOOR, chemical_space
from daikonstudio.infrastructure.tools.params import MapParams

# Enough distinct, parseable chemistry to sit either side of the floor.
_SMILES = [
    f"{'C' * (i % 7 + 1)}c1cc{'c' if i % 2 else 'n'}cc1{'O' if i % 3 else 'Cl'}" for i in range(40)
]


def _table(n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "compound_id": [f"c{i}" for i in range(n)],
            "smiles": _SMILES[:n],
            "IC50": [float(i % 11) + 0.5 for i in range(n)],
        }
    )


def test_a_set_too_small_to_lay_out_is_refused_and_told_what_to_do_instead() -> None:
    """A layout fitted on a handful of compounds draws the seed, and a reader cannot see
    that by looking at it. Refusing names the tool that answers the same question with a
    number anyone can check."""
    with pytest.raises(ValueError, match="at least 25"):
        chemical_space(MapParams(), {"compounds": _table(MAP_FLOOR - 1)})


def test_every_row_keeps_its_place_and_the_seed_is_part_of_the_result() -> None:
    frame = _table(30)
    out = chemical_space(MapParams(colour_by="IC50", seed=7), {"compounds": frame})
    placed = out["map"]
    # Row order is the caller's: a later step may join on position.
    assert placed["compound_id"].to_list() == frame["compound_id"].to_list()
    assert placed["x"].null_count() == 0 and placed["y"].null_count() == 0
    assert out["meta"]["n_placed"] == 30
    assert out["meta"]["coloured_by"] == "IC50"
    assert out["meta"]["seed"] == 7


def test_a_colour_column_that_is_not_there_is_reported_rather_than_assumed() -> None:
    """Silently colouring by nothing would read as a set with no activity in it."""
    out = chemical_space(MapParams(colour_by="pIC50"), {"compounds": _table(30)})
    assert out["meta"]["coloured_by"] is None
    assert out["meta"]["requested_colour_by"] == "pIC50"


def test_an_unparseable_structure_is_a_row_without_coordinates_not_a_crash() -> None:
    frame = _table(30)
    frame[1, "smiles"] = "not-a-molecule"
    out = chemical_space(MapParams(), {"compounds": frame})
    placed = out["map"]
    assert placed.height == 30 and out["meta"]["n_unparsed"] == 1
    assert placed["x"][1] is None and placed["y"][1] is None
    assert placed["x"].null_count() == 1
