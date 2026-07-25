from daikonstudio.infrastructure.storage.fsspec_blob_store import FsspecBlobStore


def test_put_get_roundtrip(tmp_path):
    store = FsspecBlobStore(f"file://{tmp_path}")
    uri = store.put_bytes("ws/datasets/abc/snapshot.parquet", b"payload")
    assert store.get_bytes("ws/datasets/abc/snapshot.parquet") == b"payload"
    assert uri.endswith("ws/datasets/abc/snapshot.parquet")


def test_exists_and_delete(tmp_path):
    store = FsspecBlobStore(f"file://{tmp_path}")
    assert store.exists("missing") is False
    store.put_bytes("present", b"x")
    assert store.exists("present") is True
    store.delete("present")
    assert store.exists("present") is False
