"""The shared Lightning helpers that need neither torch nor lightning to be tested."""

from __future__ import annotations

from pathlib import Path

import pytest

from daikonstudio.application.engines.checkpoints import Checkpoints
from daikonstudio.infrastructure.engines._lightning import saved_training_state
from tests.fakes.blob_store import InMemoryBlobStore


def test_a_scratch_directory_that_cannot_be_written_trains_from_the_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full TMPDIR is a reason to lose the saved progress, not the run."""
    state = Checkpoints(InMemoryBlobStore(), "ws/datasets/d/runs/r/checkpoints/").scoped(
        "lightning"
    )
    state.save("training-state", b"saved")

    def full(self: Path, data: bytes) -> int:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_bytes", full)

    assert saved_training_state(state, tmp_path, module=object()) is None
