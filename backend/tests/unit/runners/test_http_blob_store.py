"""`HttpBlobStore.put_bytes` over `httpx.MockTransport` -- no studio server, no database."""

from __future__ import annotations

import uuid

import httpx
import pytest

from daikonstudio.application.execution.failure_message import user_facing_error
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.runner.ports import HttpBlobStore, RunnerApiClient


def _store(status: int) -> HttpBlobStore:
    client = RunnerApiClient(
        "http://studio",
        "token",
        uuid.uuid4(),
        sync_transport=httpx.MockTransport(lambda request: httpx.Response(status)),
    )
    return HttpBlobStore(client)


def test_an_upload_over_the_limit_says_how_big_it_was_and_what_to_do() -> None:
    with pytest.raises(ValidationError) as caught:
        _store(413).put_bytes("w/protocols/p/artifact/model.joblib", b"x" * 2_000_000)

    message = user_facing_error(caught.value)
    assert "2 MB" in message
    assert "STUDIO_RUNNER_UPLOAD_MAX_BYTES" in message


def test_any_other_failure_is_still_an_http_error() -> None:
    with pytest.raises(httpx.HTTPStatusError):
        _store(500).put_bytes("w/uploads/x.csv", b"data")
