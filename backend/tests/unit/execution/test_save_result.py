"""`_save_result`: a finished fit's in-progress training state is freed only once the
fit's own result is safely saved -- otherwise the next attempt would have nothing at all
to resume from."""

from __future__ import annotations

import pytest

from daikonstudio.application.engines.checkpoints import (
    TRAINING_STATE,
    TRAINING_STATE_SCOPE,
    Checkpoints,
)
from daikonstudio.application.engines.context import TrainResult
from daikonstudio.application.execution.train_protocol import _save_result
from tests.fakes.blob_store import InMemoryBlobStore

ROOT = "ws/datasets/d/runs/r/checkpoints/"


def test_the_training_state_is_kept_when_the_result_did_not_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = Checkpoints(InMemoryBlobStore(), ROOT).scoped("model")
    state = stage.scoped(TRAINING_STATE_SCOPE)
    state.save(TRAINING_STATE, b"in progress")

    monkeypatch.setattr(Checkpoints, "save_result", lambda self, result: False)
    _save_result(stage, TrainResult(artifact=b"model", metrics={}))

    assert state.load(TRAINING_STATE) == b"in progress"


def test_the_training_state_is_freed_once_the_result_saved() -> None:
    stage = Checkpoints(InMemoryBlobStore(), ROOT).scoped("model")
    state = stage.scoped(TRAINING_STATE_SCOPE)
    state.save(TRAINING_STATE, b"in progress")

    _save_result(stage, TrainResult(artifact=b"model", metrics={}))

    assert state.load(TRAINING_STATE) is None
    assert stage.load("result") is not None
