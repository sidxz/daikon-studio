"""`read_compound_ids` reads a dataset's identifier column out of its snapshot.

The map asks on every hover and the scorecard on every read, so what it costs is
part of its contract, not an implementation detail: it must read the structures
asked for and not the whole snapshot.
"""

import io
import uuid

import polars as pl

from daikonstudio.application.data.compound_ids import read_compound_ids
from daikonstudio.application.data.snapshot import snapshot_key
from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.data.target import TargetKind, TargetSpec
from daikonstudio.domain.data.validation import ValidationReport


class FakeBlobStore:
    """An in-memory BlobStore double, as in `test_snapshot.py`."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def put_bytes(self, key: str, data: bytes) -> str:
        self.blobs[key] = data
        return f"memory://{key}"

    def get_bytes(self, key: str) -> bytes:
        return self.blobs[key]

    def exists(self, key: str) -> bool:
        return key in self.blobs

    def delete(self, key: str) -> None:
        del self.blobs[key]


def _dataset(id_column: str | None = "name") -> Dataset:
    return Dataset(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        name="a dataset",
        structure_column="smiles",
        target=TargetSpec(column="y", kind=TargetKind.NUMERIC),
        split=SplitSpec(strategy=SplitStrategy.RANDOM, seed=7),
        content_hash="deadbeef",
        snapshot_uri="file:///snapshot.parquet",
        row_count=3,
        validation_report=ValidationReport(total_rows=3, valid_rows=3),
        id_column=id_column,
    )


def _store_with(frame: pl.DataFrame, dataset: Dataset) -> FakeBlobStore:
    store = FakeBlobStore()
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    store.put_bytes(snapshot_key(dataset.workspace_id, dataset.id), buffer.getvalue())
    return store


def test_only_the_requested_structures_come_back() -> None:
    """The whole point of the parameter: a map hover wants a handful of IDs out of a
    snapshot that may hold hundreds of thousands, and must not pay for the rest."""
    dataset = _dataset()
    frame = pl.DataFrame(
        {"smiles": ["CCO", "CCN", "CCC"], "y": [1.0, 2.0, 3.0], "name": ["a", "b", "c"]}
    )
    store = _store_with(frame, dataset)

    ids = read_compound_ids(store, dataset, ["CCO", "CCC"])

    assert ids == {"CCO": "a", "CCC": "c"}


def test_no_identifier_column_reads_nothing_at_all() -> None:
    dataset = _dataset(id_column=None)
    store = FakeBlobStore()

    assert read_compound_ids(store, dataset, ["CCO"]) is None
    assert store.blobs == {}, "a dataset with no identifier column must not read its snapshot"


def test_asking_for_nothing_reads_nothing() -> None:
    dataset = _dataset()
    store = FakeBlobStore()

    assert read_compound_ids(store, dataset, []) == {}
    assert store.blobs == {}


# The three pins below cover `id_text`, which the review found unprotected: its
# `"" -> null` branch could be deleted and the suite would stay green, and the
# integer case was never exercised because the upload path reads every column as
# text, so a legacy snapshot's Int64 column had no test at all.


def test_an_integer_identifier_column_reads_as_digits_without_a_decimal_point() -> None:
    dataset = _dataset(id_column="num")
    frame = pl.DataFrame({"smiles": ["CCO", "CCN"], "num": [12, 7]})

    ids = read_compound_ids(_store_with(frame, dataset), dataset, ["CCO", "CCN"])

    assert ids == {"CCO": "12", "CCN": "7"}


def test_a_text_identifier_keeps_its_leading_zeros() -> None:
    dataset = _dataset()
    frame = pl.DataFrame({"smiles": ["CCO"], "name": ["007"]})

    ids = read_compound_ids(_store_with(frame, dataset), dataset, ["CCO"])

    assert ids == {"CCO": "007"}


def test_a_blank_or_whitespace_identifier_is_no_identifier() -> None:
    """A compound with an empty ID cell must come back without one, rather than
    with an empty string that renders as a blank badge."""
    dataset = _dataset()
    frame = pl.DataFrame({"smiles": ["CCO", "CCN", "CCC"], "name": ["", "  ", "real"]})

    ids = read_compound_ids(_store_with(frame, dataset), dataset, ["CCO", "CCN", "CCC"])

    assert ids == {"CCC": "real"}
