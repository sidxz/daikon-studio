"""`HttpRunRepository` over `httpx.MockTransport`: which failures are the studio being
unreachable (passing, for a training run) and which are refusals (not)."""

from __future__ import annotations

import uuid
from collections.abc import Callable

import httpx
import pytest

from daikonstudio.domain.shared.errors import ServiceUnavailableError
from daikonstudio.infrastructure.runner.ports import HttpRunRepository, RunnerApiClient


def _runs(handler: Callable[[httpx.Request], httpx.Response]) -> HttpRunRepository:
    client = RunnerApiClient(
        "http://studio", "token", uuid.uuid4(), async_transport=httpx.MockTransport(handler)
    )
    return HttpRunRepository(client)


def _refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


async def test_a_studio_that_cannot_be_reached_is_unavailable() -> None:
    with pytest.raises(ServiceUnavailableError):
        await _runs(_refused).get_by_id(uuid.uuid4())


async def test_a_gateway_with_no_studio_behind_it_is_unavailable() -> None:
    with pytest.raises(ServiceUnavailableError):
        await _runs(lambda request: httpx.Response(503)).get_by_id(uuid.uuid4())


async def test_a_refusal_is_not_mistaken_for_unavailability() -> None:
    """403: another runner owns the run now. Training on would be wasted work."""
    with pytest.raises(httpx.HTTPStatusError):
        await _runs(lambda request: httpx.Response(403)).get_by_id(uuid.uuid4())
