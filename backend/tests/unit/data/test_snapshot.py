import polars as pl

from daikonstudio.application.data.snapshot import write_snapshot


class FakeBlobStore:
    """An in-memory BlobStore double -- exercises write_snapshot's contract without
    touching the filesystem; the fsspec-backed implementation has its own tests."""

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


def frame() -> pl.DataFrame:
    return pl.DataFrame({"smiles": ["CCO", "CCN"], "y": [1.0, 2.0]})


def test_write_snapshot_writes_to_the_expected_key_and_returns_its_uri():
    store = FakeBlobStore()
    uri, _content_hash = write_snapshot(store, "ws-1", "ds-1", frame())
    assert uri == "memory://ws-1/datasets/ds-1/snapshot.parquet"
    assert store.exists("ws-1/datasets/ds-1/snapshot.parquet")


def test_write_snapshot_hash_is_deterministic_for_identical_content():
    store = FakeBlobStore()
    _uri1, hash1 = write_snapshot(store, "ws-1", "ds-1", frame())
    _uri2, hash2 = write_snapshot(store, "ws-1", "ds-2", frame())
    assert hash1 == hash2


def test_write_snapshot_hash_changes_when_content_differs():
    store = FakeBlobStore()
    _uri1, hash1 = write_snapshot(store, "ws-1", "ds-1", frame())
    other = pl.DataFrame({"smiles": ["CCO", "CCN"], "y": [1.0, 3.0]})
    _uri2, hash2 = write_snapshot(store, "ws-1", "ds-2", other)
    assert hash1 != hash2


def test_written_bytes_round_trip_through_the_store():
    store = FakeBlobStore()
    write_snapshot(store, "ws-1", "ds-1", frame())
    round_tripped = pl.read_parquet(store.get_bytes("ws-1/datasets/ds-1/snapshot.parquet"))
    assert round_tripped.equals(frame())
