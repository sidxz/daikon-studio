"""The lane executor: a default-lane tool runs here, any other becomes a queued Run."""

from __future__ import annotations

import asyncio
import copy
import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from daikonstudio.application.ports.tool_executor import ToolInput, ToolOutput, ToolRunContext
from daikonstudio.application.tools.tool_run import OUTPUTS_MANIFEST, ToolRunParams
from daikonstudio.domain.execution.run import Run, RunKind, RunStatus
from daikonstudio.domain.runners.runner import Runner
from daikonstudio.infrastructure.tools.catalog import default_registry
from daikonstudio.infrastructure.tools.lane_executor import RUNNER_STALE_SECONDS, LaneToolExecutor
from tests.fakes.blob_store import InMemoryBlobStore


class _Runs:
    def __init__(self) -> None:
        self.rows: dict[uuid.UUID, Run] = {}

    async def add(self, run: Run) -> None:
        self.rows[run.id] = copy.deepcopy(run)

    async def update(self, run: Run) -> None:
        self.rows[run.id] = copy.deepcopy(run)

    async def get_by_id(self, run_id: uuid.UUID) -> Run | None:
        row = self.rows.get(run_id)
        return copy.deepcopy(row) if row else None


class _Enqueuer:
    """Records the lane; then, as told, plays the runner: READY with outputs, or FAILED."""

    def __init__(self, runs: _Runs, store: InMemoryBlobStore, outcome: str | None) -> None:
        self.runs, self.store, self.outcome = runs, store, outcome
        self.lanes: list[str] = []

    async def enqueue(self, run_id: uuid.UUID, lane: str = "default") -> None:
        self.lanes.append(lane)
        run = self.runs.rows[run_id]
        if self.outcome is None:
            return
        run.start()
        if self.outcome == "fail":
            run.fail("RDKit could not read the structure.")
            return
        prefix = ToolRunParams.from_params(run.params).output_prefix
        blob = f"{prefix}prepared.pdb"
        self.store.put_bytes(blob, b"ATOM")
        manifest = [{"key": "prepared", "kind": "structure", "blob_key": blob, "summary": {}}]
        self.store.put_bytes(f"{prefix}{OUTPUTS_MANIFEST}", json.dumps(manifest).encode())
        run.succeed(f"memory://{prefix}{OUTPUTS_MANIFEST}")


class _Runners:
    def __init__(self, runners: list[Runner]) -> None:
        self.runners = runners

    async def list(self) -> list[Runner]:
        return self.runners


class _Local:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def available(self, tool_id: str) -> bool:
        return tool_id == "fetch_pdb"

    async def run(
        self,
        *,
        tool_id: str,
        params: Mapping[str, Any],
        inputs: Sequence[ToolInput],
        context: ToolRunContext,
    ) -> list[ToolOutput]:
        self.calls.append(tool_id)
        return []


WS = uuid.uuid4()
CONTEXT = ToolRunContext(
    workspace_id=WS,
    requested_by=uuid.uuid4(),
    study_id=uuid.uuid4(),
    step_id=uuid.uuid4(),
    output_prefix=f"{WS}/studies/x/values/",
)


# A sentinel, not a timestamp: a time taken at import would age past the
# staleness window during a long suite and make every runner look offline.
_FRESH = object()


def _executor(
    outcome: str | None = "ready",
    lanes: tuple[str, ...] | None = ("chem-tools",),
    revoked_at: datetime | None = None,
    last_seen_at: datetime | object | None = _FRESH,
) -> tuple[LaneToolExecutor, _Runs, _Enqueuer, _Local]:
    runs, store, local = _Runs(), InMemoryBlobStore(), _Local()
    enqueuer = _Enqueuer(runs, store, outcome)
    runners = (
        []
        if lanes is None
        else [
            Runner(
                name="r",
                lanes=lanes,
                token_hash="h",
                revoked_at=revoked_at,
                last_seen_at=(
                    datetime.now(UTC) if last_seen_at is _FRESH else last_seen_at  # type: ignore[arg-type]
                ),
            )
        ]
    )
    executor = LaneToolExecutor(
        default_registry(),
        local,  # type: ignore[arg-type]
        runs,  # type: ignore[arg-type]
        enqueuer,
        _Runners(runners),  # type: ignore[arg-type]
        store,
        poll_seconds=0.01,
    )
    return executor, runs, enqueuer, local


async def _run(executor: LaneToolExecutor, tool_id: str = "prepare_structure") -> list[ToolOutput]:
    return await executor.run(tool_id=tool_id, params={}, inputs=[], context=CONTEXT)


async def test_a_default_lane_tool_runs_locally() -> None:
    executor, runs, enqueuer, local = _executor()
    assert await executor.available("fetch_pdb")
    await _run(executor, "fetch_pdb")
    assert local.calls == ["fetch_pdb"] and enqueuer.lanes == [] and runs.rows == {}


async def test_a_lane_without_a_runner_is_unavailable() -> None:
    assert not await _executor(lanes=None)[0].available("prepare_structure")
    assert not await _executor(lanes=("gpu",))[0].available("prepare_structure")


async def test_a_runner_not_seen_lately_does_not_count() -> None:
    """Registered is not serving. A runner that stopped polling would take the
    step and never return it; the step is skipped with a reason instead."""
    stale = datetime.now(UTC) - timedelta(seconds=RUNNER_STALE_SECONDS + 1)
    assert not await _executor(last_seen_at=stale)[0].available("prepare_structure")
    assert not await _executor(last_seen_at=None)[0].available("prepare_structure")


async def test_a_revoked_runner_serves_no_lane() -> None:
    executor = _executor(revoked_at=datetime.now(UTC))[0]
    assert not await executor.available("prepare_structure")


async def test_a_lane_tool_is_queued_and_its_outputs_come_back() -> None:
    executor, runs, enqueuer, local = _executor()
    assert await executor.available("prepare_structure")
    (out,) = await _run(executor)
    assert enqueuer.lanes == ["chem-tools"] and local.calls == []
    (run,) = runs.rows.values()
    assert run.kind is RunKind.TOOL and run.status is RunStatus.READY
    prefix = f"{WS}/studies/{CONTEXT.study_id}/tool-runs/{run.id}/"
    assert run.params["output_prefix"] == prefix and run.params["lane"] == "chem-tools"
    assert (out.key, out.blob_key) == ("prepared", f"{prefix}prepared.pdb")


async def test_a_failed_run_raises_the_runners_message() -> None:
    executor = _executor("fail")[0]
    with pytest.raises(RuntimeError, match="RDKit could not read the structure"):
        await _run(executor)


async def test_cancelling_the_wait_cancels_the_run() -> None:
    executor, runs, _, _ = _executor(outcome=None)
    task = asyncio.create_task(_run(executor))
    while not runs.rows:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (run,) = runs.rows.values()
    assert run.status is RunStatus.CANCELLED
