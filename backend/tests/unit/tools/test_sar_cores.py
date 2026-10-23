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


def test_naming_is_all_or_nothing_per_column() -> None:
    table = pl.DataFrame(
        {
            "id": ["a", "b", "c"],
            "decomposed": [True] * 3,
            "R1": ["C[*:1]", "Cl[*:1]", "[H][*:1]"],
            "R1_smiles": ["C[*:1]", "Cl[*:1]", "[H][*:1]"],
            "R2": ["CO[*:2]", "C1CCOCC1[*:2]", "FC(F)(F)[*:2]"],
            "R2_smiles": ["CO[*:2]", "C1CCOCC1[*:2]", "FC(F)(F)[*:2]"],
        }
    )
    named = name_substituents(None, {"table": table})["table"]
    assert named["R1"].to_list() == ["Me", "Cl", "H"]
    assert named["R2"].to_list() == table["R2"].to_list()  # one unnamed fragment keeps SMILES
    assert named["R1_smiles"].to_list() == table["R1_smiles"].to_list()
    assert name_substituents(None, {"table": named})["table"]["R1"].to_list() == ["Me", "Cl", "H"]


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
