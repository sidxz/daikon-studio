"""Unit coverage for `agent.poll_once`'s claim/execute cycle, driven entirely
over `httpx.MockTransport` -- no real studio server, no database.

`jobs.run_job` itself is monkeypatched out (its own contract -- catch/re-raise,
persist FAILED -- is `tests/unit/execution/test_jobs.py`'s job); what's under
test here is `poll_once`'s side of that contract: does it build the ctx with
the claimed deadline, does it catch a handler failure without propagating it,
does an empty claim skip execution entirely.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import pytest

from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.infrastructure import jobs
from daikonstudio.infrastructure.runner import agent
from daikonstudio.infrastructure.runner.wire import RunEnvelope


def _settings() -> agent.AgentSettings:
    return agent.AgentSettings(
        url="http://studio.test", runner_token="drt_test_token", poll_seconds=0.01
    )


def _run_envelope_json(run_id: uuid.UUID) -> dict[str, Any]:
    run = Run(
        id=run_id,
        kind=RunKind.PREDICTION,
        workspace_id=uuid.uuid4(),
        requested_by=uuid.uuid4(),
        cache_key="k",
    )
    return RunEnvelope.from_domain(run).model_dump(mode="json")


async def _poll_against(handler: Any) -> bool:
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://studio.test") as api:
        return await agent.poll_once(api, _settings())


async def test_a_204_claim_returns_false_and_runs_no_job(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[uuid.UUID] = []

    async def fake_run_job(ctx: dict[str, Any], run_id: uuid.UUID) -> None:
        calls.append(run_id)

    monkeypatch.setattr(jobs, "run_job", fake_run_job)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/runner/claim"
        return httpx.Response(204)

    executed = await _poll_against(handler)

    assert executed is False
    assert calls == []


async def test_a_200_claim_runs_the_job_with_the_claimed_deadline_and_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = uuid.uuid4()
    recorded: list[tuple[dict[str, Any], uuid.UUID]] = []

    async def fake_run_job(ctx: dict[str, Any], claimed_run_id: uuid.UUID) -> None:
        recorded.append((ctx, claimed_run_id))

    monkeypatch.setattr(jobs, "run_job", fake_run_job)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"run": _run_envelope_json(run_id), "deadline_seconds": 42}
        )

    executed = await _poll_against(handler)

    assert executed is True
    assert len(recorded) == 1
    ctx, recorded_run_id = recorded[0]
    assert recorded_run_id == run_id
    assert ctx["job_deadline_seconds"] == 42


async def test_a_job_failure_is_caught_and_poll_once_still_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = uuid.uuid4()

    async def failing_run_job(ctx: dict[str, Any], claimed_run_id: uuid.UUID) -> None:
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(jobs, "run_job", failing_run_job)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"run": _run_envelope_json(run_id), "deadline_seconds": 10}
        )

    executed = await _poll_against(handler)  # must not raise

    assert executed is True
