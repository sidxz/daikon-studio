import math
import uuid

import pytest

from daikonstudio.application.data.delete_dataset import dataset_folder
from daikonstudio.application.engines.checkpoints import (
    Checkpoints,
    checkpoint_root,
    pack_result,
    unpack_result,
)
from daikonstudio.application.engines.context import TrainResult
from tests.fakes.blob_store import InMemoryBlobStore

ROOT = "ws/datasets/d/runs/r/checkpoints/"


def test_the_root_sits_under_the_dataset_folder_so_deleting_the_dataset_deletes_it():
    ws, dataset, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    root = checkpoint_root(ws, dataset, run)
    assert root.startswith(dataset_folder(ws, dataset))
    assert root.endswith(f"runs/{run}/checkpoints/")


def test_save_then_load_round_trips_in_a_scope():
    store = InMemoryBlobStore()
    stage = Checkpoints(store, ROOT).scoped("model", engine="x", engine_version="1")
    stage.save("result", b"payload")
    assert stage.load("result") == b"payload"
    assert all(key.startswith(ROOT + "model/") for key in store.blobs)


def test_a_different_fingerprint_reads_as_nothing_saved():
    store = InMemoryBlobStore()
    Checkpoints(store, ROOT).scoped("model", engine_version="1").save("result", b"old")
    assert Checkpoints(store, ROOT).scoped("model", engine_version="2").load("result") is None


def test_a_corrupted_blob_reads_as_nothing_saved():
    store = InMemoryBlobStore()
    checkpoints = Checkpoints(store, ROOT)
    checkpoints.save("state", b"0123456789")
    (blob,) = [key for key in store.blobs if not key.endswith(".json")]
    store.blobs[blob] = b"01234"  # truncated mid-write
    assert checkpoints.load("state") is None


def test_a_save_whose_marker_write_fails_keeps_the_previous_save_loadable():
    store = InMemoryBlobStore()
    checkpoints = Checkpoints(store, ROOT)
    checkpoints.save("state", b"first")
    real_put = store.put_bytes

    def fail_on_marker(key: str, data: bytes) -> str:
        if key.endswith(".json"):
            raise OSError("disk full")
        return real_put(key, data)

    store.put_bytes = fail_on_marker  # type: ignore[method-assign]
    checkpoints.save("state", b"second")  # must not raise
    store.put_bytes = real_put  # type: ignore[method-assign]
    assert checkpoints.load("state") == b"first"


def test_a_failing_store_never_raises_from_save_or_load():
    class Broken(InMemoryBlobStore):
        def put_bytes(self, key: str, data: bytes) -> str:
            raise OSError("413: upload exceeds runner_upload_max_bytes")

        def get_bytes(self, key: str) -> bytes:
            raise OSError("connection reset")

    checkpoints = Checkpoints(Broken(), ROOT)
    checkpoints.save("state", b"x")
    assert checkpoints.load("state") is None


def test_repeated_saves_alternate_two_slots_so_storage_stays_bounded():
    store = InMemoryBlobStore()
    checkpoints = Checkpoints(store, ROOT)
    for epoch in range(5):
        checkpoints.save("state", f"epoch {epoch}".encode())
        assert checkpoints.load("state") == f"epoch {epoch}".encode()
    assert sorted(key for key in store.blobs if not key.endswith(".json")) == [
        ROOT + "state.a",
        ROOT + "state.b",
    ]


@pytest.mark.parametrize("name", ["a/b", "..", ".", "", "x y"])
def test_unsafe_names_are_refused(name):
    with pytest.raises(ValueError):
        Checkpoints(InMemoryBlobStore(), ROOT).scoped(name)


def test_clear_deletes_everything_under_the_root_only():
    store = InMemoryBlobStore()
    store.put_bytes("ws/datasets/d/snapshot.parquet", b"keep")
    checkpoints = Checkpoints(store, ROOT)
    checkpoints.scoped("model").save("result", b"x")
    checkpoints.clear()
    assert list(store.blobs) == ["ws/datasets/d/snapshot.parquet"]


def test_a_train_result_round_trips_exactly():
    result = TrainResult(
        artifact=b"\x00model\xff",
        metrics={"y": {"mcc": 0.5, "auroc": float("nan")}},
        validation_metrics=None,
        cutoffs={"y": 0.31},
    )
    restored = unpack_result(pack_result(result))
    assert restored.artifact == result.artifact
    assert restored.metrics["y"]["mcc"] == 0.5 and math.isnan(restored.metrics["y"]["auroc"])
    assert restored.validation_metrics is None
    assert restored.cutoffs == {"y": 0.31}
    assert unpack_result(pack_result(TrainResult(artifact=b"", metrics={}))).cutoffs is None


def test_an_unreadable_marker_does_not_block_later_saves():
    store = InMemoryBlobStore()
    stage = Checkpoints(store, ROOT).scoped("model")
    store.put_bytes(ROOT + "model/result.json", b"{ torn")

    stage.save("result", b"payload")

    assert stage.load("result") == b"payload"


def test_a_marker_that_is_not_an_object_reads_as_nothing_saved():
    store = InMemoryBlobStore()
    stage = Checkpoints(store, ROOT).scoped("model")
    store.put_bytes(ROOT + "model/result.json", b"[1, 2]")

    assert stage.load("result") is None
    stage.save("result", b"payload")
    assert stage.load("result") == b"payload"


def test_discard_empties_both_slots_and_the_marker_whatever_the_fingerprint():
    store = InMemoryBlobStore()
    saved = Checkpoints(store, ROOT).scoped("lightning", torch="2.1")
    saved.save("training-state", b"first")
    saved.save("training-state", b"second")  # both slots now hold something

    # A scope built without the fingerprint the state was saved under, as the caller
    # that frees it has none to give.
    Checkpoints(store, ROOT).scoped("lightning").discard("training-state")

    assert saved.load("training-state") is None
    names = ("training-state.a", "training-state.b", "training-state.json")
    assert [store.blobs[f"{ROOT}lightning/{name}"] for name in names] == [b"", b"", b""]


def test_discard_writes_nothing_for_a_name_that_was_never_saved():
    store = InMemoryBlobStore()

    Checkpoints(store, ROOT).scoped("lightning").discard("training-state")

    assert store.blobs == {}


def test_a_save_after_a_discard_is_loadable():
    store = InMemoryBlobStore()
    stage = Checkpoints(store, ROOT).scoped("lightning")
    stage.save("training-state", b"old")
    stage.discard("training-state")

    stage.save("training-state", b"new")

    assert stage.load("training-state") == b"new"


def test_a_result_that_cannot_be_packed_is_not_saved_and_does_not_raise(monkeypatch):
    import daikonstudio.application.engines.checkpoints as module

    def explode(result):
        raise MemoryError("too large")

    monkeypatch.setattr(module, "pack_result", explode)
    store = InMemoryBlobStore()

    saved = (
        Checkpoints(store, ROOT)
        .scoped("model")
        .save_result(TrainResult(artifact=b"x", metrics={}))
    )

    assert saved is False
    assert store.blobs == {}
