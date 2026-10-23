"""Series, cores, the decomposed SAR table and substituent names."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pytest

from daikonstudio.infrastructure.tools.impl.cores import (
    cluster_series,
    decompose_rgroups,
    find_cores,
    name_substituents,
    score_core,
)
from daikonstudio.infrastructure.tools.params import CoreParams

DATA = Path(__file__).parent / "data"
BENZAMIDES = pl.read_csv(DATA / "benzamides.csv")
BENZAZOLES = pl.read_csv(DATA / "benzazoles.csv")
MIXED = pl.DataFrame(
    {  # the sibling's three chemotypes: benzamides, 4-anilinoquinazolines, 2-substituted indoles
        "ID": [f"C{n:02d}" for n in range(15)],
        "SMILES": [
            "O=C(Nc1ccc(OC)cc1)c1ccc(C)cc1",
            "O=C(Nc1ccc(Cl)cc1)c1ccc(C)cc1",
            "O=C(Nc1ccc(F)cc1)c1ccc(C)cc1",
            "O=C(Nc1ccc(C)cc1)c1ccc(C)cc1",
            "O=C(Nc1ccc(Br)cc1)c1ccc(C)cc1",
            "Clc1ccc(Nc2ncnc3ccccc23)cc1",
            "Fc1ccc(Nc2ncnc3ccccc23)cc1",
            "COc1ccc(Nc2ncnc3ccccc23)cc1",
            "Cc1ccc(Nc2ncnc3ccccc23)cc1",
            "CCc1ccc(Nc2ncnc3ccccc23)cc1",
            "Cc1cc2cc(Cl)ccc2[nH]1",
            "Cc1cc2cc(OC)ccc2[nH]1",
            "Cc1cc2cc(F)ccc2[nH]1",
            "Cc1cc2cc(Br)ccc2[nH]1",
            "Cc1cc2cc(C)ccc2[nH]1",
        ],
        "pIC50": [5 + 0.1 * n for n in range(15)],
    }
)
AMIDE_CORE = "O=C(Nc1ccccc1)c1ccccc1"


def test_benzazoles_share_one_core_with_one_x_position() -> None:
    out = find_cores(None, {"compounds": BENZAZOLES})["candidates"]
    assert out["meta"]["n_compounds"] == 16
    assert [c["rank"] for c in out["candidates"]] == list(range(1, len(out["candidates"]) + 1))
    best = next(c for c in out["candidates"] if c["n_variable"] == 1)
    assert set(best["variable"]["X1"].split("/")) == {"C", "N", "O", "S"}
    assert best["coverage"] == 1.0 and best["svg"].startswith("<?xml")


def test_a_hand_typed_core_is_scored_not_filtered() -> None:
    out = score_core(CoreParams(core=AMIDE_CORE), {"compounds": BENZAMIDES})["candidates"]
    (cand,) = out["candidates"]
    assert cand["rank"] == 1
    assert set(cand["unmatched"]) == {"CPD-004", "CPD-011"}
    assert {m["id"] for m in cand["misses"]} == {"CPD-004", "CPD-011"} and all(
        m["reason"] for m in cand["misses"]
    )
    assert 0 < cand["coverage"] < 1


def test_a_core_that_does_not_parse_fails_the_step() -> None:
    with pytest.raises(ValueError):
        score_core(CoreParams(core="not[a core"), {"compounds": BENZAMIDES})


def test_decomposition_keeps_misses_hydrogens_and_geminal_pairs() -> None:
    core = {"meta": {"rank": 1}, "smarts": "c1ccccc1C"}  # toluene: one ring, one benzylic carbon
    frame = pl.DataFrame(
        {
            "ID": ["a", "b", "c", "d"],
            "SMILES": ["Cc1ccccc1", "CC(C)c1ccccc1", "CC(CC)c1ccccc1", "c1ccncc1"],
        }
    )
    table = decompose_rgroups(None, {"compounds": frame, "core": core})["table"]
    assert table.columns[0] == "id"
    assert table.filter(pl.col("id") == "d")["decomposed"][0] is False
    assert table["decomposed"].sum() == 3
    r_cols = [c for c in table.columns if re.fullmatch(r"R\d+", c)]
    assert len(r_cols) >= 2  # the gem-disubstituted benzylic carbon gets two columns
    a = table.filter(pl.col("id") == "a")
    assert any("[H]" in str(a[c][0]) for c in r_cols)  # hydrogen is a value, not a blank
    assert all(f"{c}_smiles" in table.columns for c in r_cols)


def test_a_core_matching_nothing_decomposes_nothing_quietly() -> None:
    table = decompose_rgroups(None, {"compounds": BENZAMIDES, "core": {"smarts": "c1ccsc1"}})[
        "table"
    ]
    assert table.height == BENZAMIDES.height and not table["decomposed"].any()


def test_a_group_with_no_name_is_drawn_and_a_named_one_is_not() -> None:
    fragments = ["C[*:1]", "Cl[*:1]", "C1CN(CCO[*:1])CCO1"]
    table = pl.DataFrame(
        {
            "id": ["a", "b", "c"],
            "decomposed": [True] * 3,
            "R1": fragments,
            "R1_smiles": fragments,
        }
    )
    named = name_substituents(None, {"table": table})["table"]
    # The named cells are still named although the column holds an unnameable
    # group: deciding per column is what kept whole columns as SMILES.
    assert named["R1"].to_list() == ["Me", "Cl", fragments[2]]
    svg = named["R1_svg"].to_list()
    assert svg[:2] == [None, None] and "<svg" in str(svg[2])
    assert named["R1_smiles"].to_list() == fragments
    assert (
        name_substituents(None, {"table": named})["table"]["R1"].to_list() == named["R1"].to_list()
    )


def test_a_mixed_set_is_split_and_a_single_series_is_left_alone() -> None:
    mixed = cluster_series(None, {"compounds": MIXED})["series"]
    assert mixed["series"].drop_nulls().n_unique() == 3
    assert mixed["split_recommended"][0] is True
    one = cluster_series(None, {"compounds": BENZAMIDES})["series"]
    assert one["split_recommended"][0] is False
    assert one.columns[:1] == ["id"] and {
        "whole_set_core_atoms",
        "typical_series_core_atoms",
    } <= set(one.columns)


def test_find_cores_on_an_unclustered_series_table_searches_once() -> None:
    series = BENZAMIDES.rename({"ID": "id"}).with_columns(
        pl.lit(None, dtype=pl.Int64).alias("series")
    )
    out = find_cores(None, {"compounds": BENZAMIDES, "series": series})["candidates"]
    assert out["meta"]["series"] == [] and all(c["series"] is None for c in out["candidates"])


TOLUENE = {"smarts": "c1ccccc1C"}


def test_a_repeated_compound_id_is_refused_rather_than_joined() -> None:
    frame = pl.DataFrame(
        {
            "ID": ["a", "a", "b"],
            "SMILES": ["Cc1ccccc1", "CCc1ccccc1", "CC(C)c1ccccc1"],
            "pIC50": [5.0, 6.0, 7.0],
        }
    )
    with pytest.raises(ValueError, match="repeats: a"):
        decompose_rgroups(None, {"compounds": frame, "core": TOLUENE})
    with pytest.raises(ValueError, match="repeats: a"):
        cluster_series(None, {"compounds": frame})


def test_an_uploaded_rgroup_column_does_not_shadow_the_decomposition() -> None:
    frame = pl.DataFrame(
        {
            "ID": ["a", "b"],
            "SMILES": ["Cc1ccccc1", "CCc1ccccc1"],
            "R1": ["whatever the chemist typed", "and here too"],
        }
    )
    table = decompose_rgroups(None, {"compounds": frame, "core": TOLUENE})["table"]
    assert table["R1_input"].to_list() == frame["R1"].to_list()
    assert all("[" in str(v) for v in table["R1"].to_list())  # the fragment SMILES


def test_a_table_that_already_has_an_id_column_still_decomposes() -> None:
    frame = pl.DataFrame(
        {
            "Name": ["first", "second"],
            "SMILES": ["Cc1ccccc1", "CCc1ccccc1"],
            "id": ["x", "y"],
            "pIC50": [5.0, 6.0],
        }
    )
    table = decompose_rgroups(None, {"compounds": frame, "core": TOLUENE})["table"]
    assert table["id"].to_list() == ["first", "second"]
    assert table["id_input"].to_list() == ["x", "y"]
    assert table["decomposed"].all()


def test_a_series_table_that_does_not_recommend_a_split_is_searched_whole() -> None:
    series = (
        MIXED.rename({"ID": "id"})
        .with_columns(
            pl.Series("series", [1] * 8 + [2] * 7, dtype=pl.Int64),
            pl.lit(False).alias("split_recommended"),
        )
        .select(["id", "series", "split_recommended"])
    )
    whole = find_cores(None, {"compounds": MIXED, "series": series})["candidates"]
    assert whole["meta"]["series"] == []
    assert all(c["series"] is None for c in whole["candidates"])
    split = find_cores(
        None,
        {
            "compounds": MIXED,
            "series": series.with_columns(pl.lit(True).alias("split_recommended")),
        },
    )["candidates"]
    assert split["meta"]["series"] == [1, 2]


def test_a_candidates_description_carries_the_rank_it_is_shown_under() -> None:
    series = (
        MIXED.rename({"ID": "id"})
        .with_columns(
            pl.Series("series", [1] * 8 + [2] * 7, dtype=pl.Int64),
            pl.lit(True).alias("split_recommended"),
        )
        .select(["id", "series", "split_recommended"])
    )
    out = find_cores(None, {"compounds": MIXED, "series": series})["candidates"]
    for cand in out["candidates"]:
        heading = re.search(r"#(\d+)", cand["description"][0])
        assert heading is not None and int(heading.group(1)) == cand["rank"]


def test_a_candidate_description_never_carries_its_smarts() -> None:
    out = find_cores(None, {"compounds": BENZAMIDES})["candidates"]
    assert out["candidates"]
    for cand in out["candidates"]:
        assert not any("[#" in line for line in cand["description"])
    assert len(out["meta"]["notes"]) <= 5
    (scored,) = score_core(CoreParams(core=AMIDE_CORE), {"compounds": BENZAMIDES})["candidates"][
        "candidates"
    ]
    assert not any("[#" in line for line in scored["description"])


def test_an_arbitrary_embedding_is_flagged_on_the_row() -> None:
    table = decompose_rgroups(None, {"compounds": BENZAMIDES, "core": {"smarts": AMIDE_CORE}})[
        "table"
    ]
    assert "ambiguous" in table.columns and not table["ambiguous"].any()


def test_an_uploaded_series_column_is_kept_under_its_own_name() -> None:
    frame = BENZAMIDES.with_columns(pl.lit("the chemist's own grouping").alias("series"))
    out = cluster_series(None, {"compounds": frame})["series"]
    assert out["series_input"].unique().to_list() == ["the chemist's own grouping"]
    assert out["series"].dtype == pl.Int64
