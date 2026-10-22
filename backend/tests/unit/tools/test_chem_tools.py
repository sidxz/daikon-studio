import polars as pl
import pytest

from daikonstudio.infrastructure.tools.impl.chem import (
    PROPERTY_COLUMNS,
    property_confound,
    property_profile,
    standardize,
    structural_alerts,
    structure_column,
)
from daikonstudio.infrastructure.tools.params import ActivityParams, ConfoundParams

FRAME = pl.DataFrame(
    {
        "ID": ["a", "b", "c", "d"],
        "SMILES": ["O=C(Nc1ccccc1)c1ccccc1", "CC(=O)[O-].[Na+]", "O=C1C=CC(=O)C=C1", "notasmiles"],
        "pIC50": [6.5, 5.0, 7.2, None],
    }
)


def test_structure_column() -> None:
    assert structure_column(FRAME) == "SMILES"
    with pytest.raises(ValueError, match="ID"):
        structure_column(pl.DataFrame({"ID": ["a", "b"]}))


def test_standardize() -> None:
    out = standardize(None, {"compounds": FRAME})["compounds"]
    assert out.height == 4
    b = out.filter(pl.col("ID") == "b").row(0, named=True)
    assert b["canonical_smiles"] == "CC(=O)O" and b["changed"] is True
    d = out.filter(pl.col("ID") == "d").row(0, named=True)
    assert d["canonical_smiles"] is None and d["parse_error"]
    assert out["duplicates"].to_list() == [1, 1, 1, 1]


def test_standardize_collapses_duplicates() -> None:
    frame = pl.DataFrame({"SMILES": ["CCO", "OCC", "CCN"]})
    out = standardize(None, {"compounds": frame})["compounds"]
    assert out.height == 2
    assert out["duplicates"].to_list() == [2, 1]


def test_property_profile() -> None:
    out = property_profile(ActivityParams(activity_column="pIC50"), {"compounds": FRAME})[
        "properties"
    ]
    a = out.row(0, named=True)
    assert a["heavy_atoms"] == 15
    assert abs(a["lipophilic_efficiency"] - (6.5 - a["logp"])) < 1e-6
    assert out.row(3, named=True)["mol_weight"] is None


# Activity rises with chain length and aromatic bulk, so logP should come out
# as the strongest correlate. Eight rows, because a percentile bootstrap over
# three is not an interval anyone should read.
SERIES = pl.DataFrame(
    {
        "SMILES": [
            "CCO",
            "CCCCO",
            "CCCCCCO",
            "Oc1ccccc1",
            "CCc1ccccc1",
            "CCCCc1ccccc1",
            "c1ccc2ccccc2c1",
            "CCCCCCCCc1ccccc1",
        ],
        "pIC50": [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
    }
)


def test_property_confound_leaves_censored_readings_out_and_says_so() -> None:
    censored = SERIES.with_columns(
        pl.Series("pIC50", ["4.0", "4.5", "5.0", "5.5", "6.0", "6.5", "7.0", ">7.5"])
    )
    out = property_confound(ConfoundParams(activity_column="pIC50"), {"compounds": censored})[
        "correlation"
    ]
    assert out["meta"]["excluded"] == 1
    assert out["meta"]["n"] == 7


def test_property_confound() -> None:
    out = property_confound(ConfoundParams(activity_column="pIC50"), {"compounds": SERIES})[
        "correlation"
    ]
    assert [p["property"] for p in out["properties"]] == list(PROPERTY_COLUMNS)
    for p in out["properties"]:
        assert p["n"] == 8
        assert -1.0 <= p["r"] <= 1.0
        assert p["ci_low"] <= p["ci_high"]
    assert out["meta"]["n"] == 8
    by_name = {p["property"]: p for p in out["properties"]}
    assert by_name["logp"]["r"] > 0.9
    assert out["meta"]["strongest"]["r"] > 0.9


def test_structural_alerts() -> None:
    alerts = structural_alerts(None, {"compounds": FRAME})["alerts"]
    rows = {a["row"] for a in alerts}
    assert 2 in rows and 0 not in rows
    quinone = next(a for a in alerts if a["row"] == 2)["alerts"]
    assert any(hit["catalog"] == "PAINS" for hit in quinone)
    assert not any(" " in hit["catalog"] for a in alerts for hit in a["alerts"])


def test_structure_column_decides_on_a_sample() -> None:
    frame = pl.DataFrame({"notes": ["a remark"] * 300, "SMILES": ["CCO"] * 300})
    assert structure_column(frame) == "SMILES"
