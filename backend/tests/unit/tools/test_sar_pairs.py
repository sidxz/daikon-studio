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
    assert ab.height == 1 and ab["delta"][0] is None
    ac = pairs.filter(pl.col("id_better") == "c").filter(pl.col("id_worse") == "a")
    assert ac.height == 1 and ac["delta"][0] >= 1.5 and ac["bounded"][0] is True
    assert ac["id"][0] == "a>c"


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
    assert row["mean_delta"] == pytest.approx(-0.25)  # Me->Cl +1.0 and Cl->Me +0.5
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
