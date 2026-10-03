"""Unit coverage for `agent.poll_once`'s claim/execute cycle, driven entirely
over `httpx.MockTransport` -- no real studio server, no database.

`jobs.run_job` itself is monkeypatched out (its own contract -- catch/re-raise,
persist FAILED -- is `tests/unit/execution/test_jobs.py`'s job); what's under
test here is `poll_once`'s side of that contract: does it build the ctx with
the claimed deadline, does it catch a handler failure without propagating it,
does an empty claim skip execution entirely.
"""

from __future__ import annotations

import asyncio
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
            200,
            json={"run": _run_envelope_json(run_id), "deadline_seconds": 42, "lease_seconds": 30},
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
            200,
            json={"run": _run_envelope_json(run_id), "deadline_seconds": 10, "lease_seconds": 30},
        )

    executed = await _poll_against(handler)  # must not raise

    assert executed is True


async def test_poll_once_heartbeats_a_long_job_and_stops_when_it_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Important 1+2, final review -- the lease is renewed only by run-scoped
    HTTP calls, which during a fit come only from `ctx.report` (and only
    chemprop's engine calls that at all). `poll_once` must heartbeat on its
    own timer so a long, healthy job on any engine doesn't lose its claim."""
    run_id = uuid.uuid4()
    heartbeat_calls = 0

    async def slow_run_job(ctx: dict[str, Any], claimed_run_id: uuid.UUID) -> None:
        await asyncio.sleep(0.5)  # several multiples of the 1/3s heartbeat interval below

    monkeypatch.setattr(jobs, "run_job", slow_run_job)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal heartbeat_calls
        if request.url.path == "/api/v1/runner/claim":
            return httpx.Response(
                200,
                json={
                    "run": _run_envelope_json(run_id),
                    "deadline_seconds": 1800,
                    "lease_seconds": 1,  # -> a heartbeat every ~0.33s
                },
            )
        assert request.url.path == f"/api/v1/runner/runs/{run_id}"
        heartbeat_calls += 1
        return httpx.Response(200, json=_run_envelope_json(run_id))

    executed = await _poll_against(handler)

    assert executed is True
    assert heartbeat_calls >= 1, "expected at least one heartbeat during the job"

    # The heartbeat task must be cancelled once the job finishes -- give it
    # several more multiples of the interval and confirm the count doesn't move.
    calls_at_job_end = heartbeat_calls
    await asyncio.sleep(0.5)
    assert heartbeat_calls == calls_at_job_end, (
        "heartbeat task kept firing after poll_once returned -- not cancelled"
    )


async def test_a_500_claim_response_returns_false_and_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[uuid.UUID] = []

    async def fake_run_job(ctx: dict[str, Any], run_id: uuid.UUID) -> None:
        calls.append(run_id)

    monkeypatch.setattr(jobs, "run_job", fake_run_job)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "studio is restarting"})

    executed = await _poll_against(handler)  # must not raise

    assert executed is False
    assert calls == []


async def test_an_unreachable_studio_returns_false_and_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[uuid.UUID] = []

    async def fake_run_job(ctx: dict[str, Any], run_id: uuid.UUID) -> None:
        calls.append(run_id)

    monkeypatch.setattr(jobs, "run_job", fake_run_job)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    executed = await _poll_against(handler)  # must not raise

    assert executed is False
    assert calls == []


async def test_a_job_past_its_deadline_and_grace_is_failed_and_the_agent_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hard kill. The heartbeat renews the lease whatever the fit does, so a
    fit that never returns would hold its run RUNNING and its runner busy
    forever. Past deadline + grace the agent records the failure and exits."""
    run_id = uuid.uuid4()

    async def hang(ctx: dict[str, Any], claimed_run_id: uuid.UUID) -> None:
        await asyncio.sleep(3600)

    failed: list[str] = []

    async def fake_fail_run(ctx: dict[str, Any], claimed_run_id: uuid.UUID, message: str) -> None:
        assert claimed_run_id == run_id
        failed.append(message)

    exits: list[int] = []
    monkeypatch.setattr(jobs, "run_job", hang)
    monkeypatch.setattr(jobs, "fail_run", fake_fail_run)
    monkeypatch.setattr(agent, "_exit", lambda code: exits.append(code))
    monkeypatch.setattr(agent, "_HARD_KILL_GRACE_SECONDS", 0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/runner/claim":
            return httpx.Response(
                200,
                json={
                    "run": _run_envelope_json(run_id),
                    "deadline_seconds": 0,
                    "lease_seconds": 600,
                },
            )
        return httpx.Response(200, json=_run_envelope_json(run_id))

    executed = await _poll_against(handler)

    assert executed is True
    assert exits == [3]
    assert failed and "stopped by the runner" in failed[0]
