import polars as pl

from daikonstudio.application.tools.codecs import decode, encode, extension


def test_table_round_trip() -> None:
    frame = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    blob, summary = encode("compound_table", frame)
    assert summary == {"rows": 3, "columns": ["a", "b"]}
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
