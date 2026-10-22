import polars as pl

from daikonstudio.application.tools.codecs import decode, encode, extension


def test_table_round_trip() -> None:
    frame = pl.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [0.5, float("nan"), float("inf")]}
    )
    blob, summary = encode("compound_table", frame)
    assert (summary["rows"], summary["columns"]) == (3, ["a", "b", "c"])
    # (dtype, filled, distinct, min, max, median) per column. A summary is stored as JSON,
    # so a NaN counts as empty and an infinity stays out of the range.
    assert {
        p["name"]: (p["dtype"], p["filled"], p["distinct"], p["min"], p["max"], p["median"])
        for p in summary["columns_profile"]
    } == {
        "a": ("Int64", 3, 3, 1, 3, 2),
        "b": ("String", 3, 3, None, None, None),
        "c": ("Float64", 2, 2, 0.5, 0.5, 0.5),
    }
    assert summary["withheld"] == []  # nothing here is a structure or a sequence
    assert decode("compound_table", blob).equals(frame)  # type: ignore[attr-defined]
    assert extension("compound_table") == "parquet"
    assert extension("structure") == "json"


def test_dict_summary_is_meta() -> None:
    value = {"format": "pdb", "text": "ATOM", "meta": {"id": "4TRO"}}
    blob, summary = encode("structure", value)
    assert summary == {"id": "4TRO"}
    assert decode("structure", blob) == value


def test_list_summary_counts_items() -> None:
    assert encode("alerts", [1, 2])[1] == {"items": 2}
    assert encode("alerts", {"x": 1})[1] == {}
