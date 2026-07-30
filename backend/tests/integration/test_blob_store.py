import fsspec.core  # type: ignore[import-untyped]
import pytest

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


def test_get_bytes_also_accepts_the_uri_put_bytes_returned(tmp_path):
    """Task 17 review, Important 3: a consumer holding a stored `*_uri` field
    (e.g. `InSilicoProtocol.artifact_uri`) must be able to read it back
    directly, not only via a freshly recomputed key -- the two can point at
    different ids once a Protocol is versioned."""
    store = FsspecBlobStore(f"file://{tmp_path}")
    uri = store.put_bytes("ws/protocols/abc/artifact/model.joblib", b"weights")

    assert store.get_bytes(uri) == b"weights"


def test_storage_options_are_forwarded_to_fsspec(monkeypatch: pytest.MonkeyPatch) -> None:
    """A MinIO or Azure endpoint reaches fsspec, or the store silently talks to the
    wrong backend -- which on S3 means a confusing 403 rather than a clear failure."""
    seen: dict[str, object] = {}

    def fake_url_to_fs(url: str, **options: object) -> tuple[object, str]:
        seen["url"] = url
        seen["options"] = options
        return (object(), url)

    monkeypatch.setattr(fsspec.core, "url_to_fs", fake_url_to_fs)

    FsspecBlobStore("s3://bucket/prefix", {"endpoint_url": "https://minio.example.edu"})

    assert seen["url"] == "s3://bucket/prefix"
    assert seen["options"] == {"endpoint_url": "https://minio.example.edu"}
