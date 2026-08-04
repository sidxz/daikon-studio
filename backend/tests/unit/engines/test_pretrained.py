"""Caching weights fetched from the internet. No network in these tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.engines._pretrained import (
    WEIGHT_SETS,
    WeightSet,
    weights_path,
)


def test_chemeleon_is_registered_with_its_published_checksum():
    weight_set = WEIGHT_SETS["CheMeleon"]
    assert weight_set.url == "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
    assert weight_set.md5 == "6a80b54fdb7de37ef0374d302f01e8ce"


def test_an_already_cached_file_is_returned_without_a_download(tmp_path, monkeypatch):
    payload = b"pretend weights"
    cached = tmp_path / "fake.pt"
    cached.write_bytes(payload)
    # setitem, not setattr: WeightSet is frozen, so swap the whole entry.
    monkeypatch.setitem(
        WEIGHT_SETS,
        "CheMeleon",
        WeightSet(url="https://unused", md5=hashlib.md5(payload).hexdigest(), filename="fake.pt"),
    )

    def _explode(*args, **kwargs):
        raise AssertionError("downloaded despite a valid cached file")

    monkeypatch.setattr("daikonstudio.infrastructure.engines._pretrained.urlretrieve", _explode)
    assert weights_path("CheMeleon", str(tmp_path)) == cached


def test_a_corrupt_download_is_rejected_and_not_left_in_the_cache(tmp_path, monkeypatch):
    """A truncated download otherwise surfaces as an unreadable-tensor error with
    no hint that the network was the cause."""

    def _write_garbage(url, filename):
        Path(filename).write_bytes(b"truncated")

    monkeypatch.setattr(
        "daikonstudio.infrastructure.engines._pretrained.urlretrieve", _write_garbage
    )
    with pytest.raises(ValidationError, match="checksum"):
        weights_path("CheMeleon", str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_an_unknown_weight_set_names_what_is_available(tmp_path):
    with pytest.raises(ValidationError, match="CheMeleon"):
        weights_path("NotAModel", str(tmp_path))
