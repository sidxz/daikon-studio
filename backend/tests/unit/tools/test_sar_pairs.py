"""Matched pairs, effects, cliffs, the additive model and the coverage gaps."""

from __future__ import annotations

import itertools
import math

import polars as pl
import pytest

from daikonstudio.infrastructure.tools.impl.pairs import (
    activity_cliffs,
    coverage_gaps,
    effect_sizes,
    free_wilson,
    matched_pairs,
)
from daikonstudio.infrastructure.tools.params import AssayParams, CliffParams


def _table(ids, r1, r2, act, extra=None):
    frame = pl.DataFrame(
        {
            "id": ids,
            "decomposed": [True] * len(ids),
            "R1": r1,
            "R1_smiles": r1,
            "R2": r2,
            "R2_smiles": r2,
            "MIC": act,
        }
    )
    return frame if extra is None else frame.with_columns(**extra)


MIC = AssayParams(activity_column="MIC", kind="concentration", higher_is_better=False, unit="uM")


def test_two_bounds_guarantee_nothing_and_a_bound_against_a_value_guarantees_a_floor() -> None:
    t = _table(
        ["a", "b", "c"],
        ["C[*:1]", "Cl[*:1]", "F[*:1]"],
        ["C[*:2]"] * 3,
        [">64", ">64", "2"],
    )
    pairs = matched_pairs(MIC, {"table": t})["pairs"]
    assert pairs.columns[0] == "id"
    ab = pairs.filter(pl.col("id").str.contains("a") & pl.col("id").str.contains("b"))
    assert ab.height == 1 and ab["delta"][0] is None and ab["ordered"][0] is False
    ac = pairs.filter(pl.col("id_better") == "c").filter(pl.col("id_worse") == "a")
    assert ac.height == 1 and ac["delta"][0] >= 1.5 and ac["bounded"][0] is True
    assert ac["id"][0] == "a>c" and ac["ordered"][0] is True


def test_direction_decides_which_compound_is_better() -> None:
    t = _table(["a", "b"], ["C[*:1]", "Cl[*:1]"], ["C[*:2]"] * 2, ["8", "2"])
    lower = matched_pairs(MIC, {"table": t})["pairs"]
    higher = matched_pairs(MIC.model_copy(update={"higher_is_better": True}), {"table": t})[
        "pairs"
    ]
    assert (lower["id_better"][0], higher["id_better"][0]) == ("b", "a")
    assert lower["delta"][0] == pytest.approx(math.log10(4))


def test_cliffs_flag_the_whole_pairs_table() -> None:
    t = _table(
        ["a", "b", "c"],
        ["C[*:1]", "Cl[*:1]", "F[*:1]"],
        ["C[*:2]"] * 3,
        ["64", "2", "32"],
    )
    pairs = matched_pairs(MIC, {"table": t})["pairs"]
    cliffs = activity_cliffs(CliffParams(min_delta=math.log10(4)), {"pairs": pairs})["cliffs"]
    assert cliffs.height == pairs.height
    assert cliffs["cliff"].to_list().count(True) == 2  # 64 vs 2 (32x) and 32 vs 2 (16x)
    assert cliffs.filter(pl.col("cliff"))["cliff_rank"].to_list() == [1, 2]


def test_effects_orient_a_change_one_way() -> None:
    pairs = pl.DataFrame(
        {
            "id": ["a>b", "d>c"],
            "position": ["R1", "R1"],
            "id_better": ["b", "c"],
            "id_worse": ["a", "d"],
            "from": ["Me", "Cl"],
            "to": ["Cl", "Me"],
            "delta": [1.0, 0.5],
            "fold": ["", ""],
            "bounded": [False, True],
            "ordered": [True, True],
            "activity_better": ["", ""],
            "activity_worse": ["", ""],
            "measurements_better": [None, None],
            "measurements_worse": [None, None],
        }
    )
    effects = effect_sizes(None, {"pairs": pairs})["effects"]
    assert effects.height == 1
    row = effects.row(0, named=True)
    assert (row["id"], row["from"], row["to"], row["n_pairs"], row["n_bounded"]) == (
        "R1 Cl>Me",
        "Cl",
        "Me",
        2,
        1,
    )
    # Me->Cl +1.0, oriented the other way; the Cl->Me +0.5 pair is bounded, so it
    # is counted but does not move the mean.
    assert row["mean_delta"] == pytest.approx(-1.0)
    assert row["direction"] == "from is better"
    assert row["ci_low"] is None  # fewer than three pairs


R1 = {"a": 0.0, "b": 1.0, "c": -0.5}
R2 = {"x": 0.0, "y": 2.0, "z": 0.5}
GRID = [
    (f"{p}{q}", p, q, 5.0 + R1[p] + R2[q])
    for p, q in itertools.product(R1, R2)
    if (p, q) != ("c", "z")
]
PIC50 = AssayParams(activity_column="pIC50", kind="log", higher_is_better=True)


def _grid_table():
    ids, r1, r2, act = zip(*GRID, strict=True)
    return pl.DataFrame(
        {
            "id": ids,
            "decomposed": [True] * len(ids),
            "R1": r1,
            "R1_smiles": r1,
            "R2": r2,
            "R2_smiles": r2,
            "pIC50": [str(a) for a in act],
        }
    )


def test_an_additive_grid_is_recovered_and_a_saturated_design_refused() -> None:
    fit = free_wilson(PIC50, {"table": _grid_table()})["fit"]
    assert fit["meta"]["ok"] and fit["meta"]["r2"] == pytest.approx(1.0)
    assert fit["meta"]["q2"] == pytest.approx(1.0)
    assert fit["meta"]["reference"] == {"R1": "a", "R2": "x"}
    coef = {(t["position"], t["group"]): t["coef"] for t in fit["terms"]}
    assert coef[("R1", "b")] == pytest.approx(1.0) and coef[("R2", "y")] == pytest.approx(2.0)
    assert fit["unmade"] == [{"groups": {"R1": "c", "R2": "z"}, "prediction": pytest.approx(5.0)}]
    small = free_wilson(PIC50, {"table": _grid_table().head(3)})["fit"]
    assert small["meta"]["ok"] is False and small["meta"]["reason"]


def test_censored_readings_are_left_out_of_the_fit_and_counted() -> None:
    table = _grid_table().with_columns(
        pl.when(pl.col("id") == "ax").then(pl.lit(">5")).otherwise(pl.col("pIC50")).alias("pIC50")
    )
    fit = free_wilson(PIC50, {"table": table})["fit"]
    assert fit["meta"]["n_excluded_censored"] == 1
    assert fit["meta"]["n_compounds"] == len(GRID) - 1


def test_gaps_are_the_unmade_combinations_most_surrounded_first() -> None:
    gaps = coverage_gaps(None, {"table": _grid_table()})["gaps"]
    assert gaps.height == 1
    assert gaps.row(0, named=True) == {
        "id": "c|z",
        "R1": "c",
        "R2": "z",
        "nearest_made": gaps["nearest_made"][0],
        "n_neighbours": 4,
    }
    assert gaps["nearest_made"][0] in {"cx", "cy", "az", "bz"}


def test_nothing_decomposed_or_no_usable_activity_is_empty_not_an_error() -> None:
    none = _grid_table().with_columns(pl.lit(False).alias("decomposed"))
    assert matched_pairs(PIC50, {"table": none})["pairs"].height == 0
    assert free_wilson(PIC50, {"table": none})["fit"]["meta"]["ok"] is False
    assert coverage_gaps(None, {"table": none})["gaps"].height == 0
    text = _grid_table().with_columns(pl.lit("n.d.").alias("pIC50"))
    assert matched_pairs(PIC50, {"table": text})["pairs"].height == 0
    fit = free_wilson(PIC50, {"table": text})["fit"]
    assert fit["meta"]["ok"] is False and "pIC50" in fit["meta"]["reason"]


def test_an_activity_column_that_is_not_there_fails_the_step() -> None:
    t = _table(["a", "b"], ["C[*:1]", "Cl[*:1]"], ["C[*:2]"] * 2, ["8", "2"])
    wrong = MIC.model_copy(update={"activity_column": "mic"})
    with pytest.raises(ValueError, match="'mic'"):
        matched_pairs(wrong, {"table": t})
    with pytest.raises(ValueError, match="'mic'"):
        free_wilson(wrong, {"table": t})


def test_a_pair_that_guarantees_nothing_asserts_no_direction() -> None:
    t = _table(["a", "b"], ["C[*:1]", "Cl[*:1]"], ["C[*:2]"] * 2, ["2", "<64"])
    row = matched_pairs(MIC, {"table": t})["pairs"].row(0, named=True)
    assert row["delta"] is None and row["fold"] is None and row["ordered"] is False
    assert row["id"] == "a|b" and ">" not in row["id"]
    assert {row["id_better"], row["id_worse"]} == {"a", "b"}


def test_a_bounded_pair_is_counted_but_does_not_move_the_mean() -> None:
    pairs = pl.DataFrame(
        {
            "position": ["R1", "R1", "R2"],
            "from": ["Cl", "Cl", "Cl"],
            "to": ["Me", "Me", "Me"],
            "delta": [1.0, 2.0, 2.0],
            "bounded": [False, True, True],
        }
    )
    effects = effect_sizes(None, {"pairs": pairs})["effects"]
    rows = {row["position"]: row for row in effects.iter_rows(named=True)}
    supported = rows["R1"]
    assert (supported["n_pairs"], supported["n_bounded"]) == (2, 1)
    assert supported["mean_delta"] == pytest.approx(1.0)
    assert supported["direction"] == "to is better"
    bounded_only = rows["R2"]
    assert bounded_only["n_pairs"] == 1 and bounded_only["n_bounded"] == 1
    assert bounded_only["mean_delta"] is None and bounded_only["ci_low"] is None
    assert bounded_only["direction"] is None


def test_a_pair_that_guarantees_nothing_is_counted_but_does_not_move_the_mean() -> None:
    pairs = pl.DataFrame(
        {
            "position": ["R1", "R1"],
            "from": ["Cl", "Cl"],
            "to": ["Me", "Me"],
            "delta": [1.0, None],
            "bounded": [False, True],
        }
    )
    row = effect_sizes(None, {"pairs": pairs})["effects"].row(0, named=True)
    # The three counts add up to n_pairs, so the reader can see the unordered pair
    # the pairs table shows beside this one.
    assert (row["n_pairs"], row["n_bounded"], row["n_unordered"]) == (2, 0, 1)
    assert row["mean_delta"] == pytest.approx(1.0)


def test_a_compound_with_a_null_position_is_left_out_rather_than_grouped_as_none() -> None:
    table = _grid_table().with_columns(
        pl.when(pl.col("id") == "ax")
        .then(pl.lit(None, dtype=pl.Utf8))
        .otherwise(pl.col("R2"))
        .alias("R2")
    )
    gaps = coverage_gaps(None, {"table": table})["gaps"]
    assert gaps.height
    assert not any("None" in str(value) for row in gaps.iter_rows() for value in row)
    fit = free_wilson(PIC50, {"table": table})["fit"]
    assert fit["meta"]["ok"] and fit["meta"]["n_excluded_incomplete"] == 1
    assert "ax" not in {fitted["id"] for fitted in fit["fitted"]}


def test_a_pair_carries_both_molecules_and_the_two_fragments_that_differ() -> None:
    """A change from one word to another says nothing to anyone who does not already know
    what the words draw as. The molecules are what a reader looks at, and the fragments are
    what tells them where to look; the fragments alone would draw the wrong picture."""
    t = _table(
        ["a", "b"],
        ["C[*:1]", "Cl[*:1]"],
        ["C[*:2]"] * 2,
        ["8", "2"],
        extra={"canonical_smiles": pl.Series(["Cc1ccccc1C", "Clc1ccccc1C"])},
    )
    pairs = matched_pairs(MIC, {"table": t})["pairs"]
    row = pairs.row(0, named=True)
    # b is the better compound, so it is the to-side and a is the from-side.
    assert (row["from_smiles"], row["to_smiles"]) == ("Cc1ccccc1C", "Clc1ccccc1C")
    assert (row["from_group_smiles"], row["to_group_smiles"]) == ("C[*:1]", "Cl[*:1]")


def test_without_a_named_structure_column_a_pair_draws_nothing_rather_than_a_fragment() -> None:
    """`structure_column` takes the first column that parses, which on a decomposed table is
    R1. A fragment parses as readily as a molecule, so guessing here would draw a picture
    that is wrong in a way nobody would question."""
    pairs = matched_pairs(
        MIC, {"table": _table(["a", "b"], ["C[*:1]", "Cl[*:1]"], ["C[*:2]"] * 2, ["8", "2"])}
    )["pairs"]
    row = pairs.row(0, named=True)
    assert row["from_smiles"] is None and row["to_smiles"] is None
    # The fragments still come through: they are the tool's own output, not a guess.
    assert row["from_group_smiles"] == "C[*:1]"
