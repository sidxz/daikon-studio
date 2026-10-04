"""The stored form of a trained model: xz-compressed, with unpacked legacy reads."""

from __future__ import annotations

import pickle

from daikonstudio.application.execution.train_protocol import pack_artifact, unpack_artifact

_XZ_MAGIC = b"\xfd7zXZ\x00"


def test_a_pickle_round_trips() -> None:
    artifact = pickle.dumps({"trees": list(range(1000))})
    assert unpack_artifact(pack_artifact(artifact)) == artifact


def test_a_zip_container_round_trips() -> None:
    artifact = b"PK\x03\x04" + b"fan-out member" * 50
    assert unpack_artifact(pack_artifact(artifact)) == artifact


def test_packed_bytes_start_with_the_xz_magic() -> None:
    assert pack_artifact(b"anything").startswith(_XZ_MAGIC)


def test_a_large_repetitive_artifact_shrinks() -> None:
    artifact = pickle.dumps([0.0] * 100_000)
    assert len(pack_artifact(artifact)) < len(artifact) / 10


def test_an_artifact_stored_before_compression_is_read_as_it_is() -> None:
    legacy_pickle = pickle.dumps({"trees": [1, 2, 3]})
    legacy_zip = b"PK\x03\x04" + b"member"
    assert unpack_artifact(legacy_pickle) == legacy_pickle
    assert unpack_artifact(legacy_zip) == legacy_zip
