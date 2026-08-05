"""Sweeps: N training runs submitted as one group, ranked together.

Against real Postgres and the real repository -- the columns exist to be
queried and grouped, and an ORM round-trip that never touches SQL would
not prove either.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.repository import (
    SqlAlchemyRunRepository,
)


@pytest_asyncio.fixture
async def sessions(_migrated_engine: AsyncEngine):
    return async_sessionmaker(_migrated_engine, expire_on_commit=False)


def _run(*, workspace_id: uuid.UUID, sweep_id: uuid.UUID | None = None, name: str = "s") -> Run:
    return Run(
        kind=RunKind.TRAINING,
        workspace_id=workspace_id,
        requested_by=uuid.uuid4(),
        cache_key="k" * 64,
        params={"name": name, "sweep_name": "BBBP comparison", "dataset_id": str(uuid.uuid4())},
        sweep_id=sweep_id,
    )


@pytest.mark.asyncio
async def test_sweep_id_and_metrics_round_trip(sessions) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    workspace_id = uuid.uuid4()
    sweep_id = uuid.uuid4()
    run = _run(workspace_id=workspace_id, sweep_id=sweep_id)
    await repository.add(run)

    run.start()
    run.record_metrics(primary_metric="mcc", value=0.603, baseline_value=0.632)
    await repository.update(run)

    stored = await repository.get(workspace_id, run.id)
    assert stored is not None
    assert stored.sweep_id == sweep_id
    assert stored.metrics == {
        "primary_metric": "mcc",
        "value": 0.603,
        "baseline_value": 0.632,
    }


@pytest.mark.asyncio
async def test_solo_run_has_no_sweep_id(sessions) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    workspace_id = uuid.uuid4()
    run = _run(workspace_id=workspace_id)
    await repository.add(run)

    stored = await repository.get(workspace_id, run.id)
    assert stored is not None
    assert stored.sweep_id is None
    assert stored.metrics is None


@pytest.mark.asyncio
async def test_list_by_sweep_returns_only_that_sweep(sessions) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    workspace_id = uuid.uuid4()
    mine, theirs = uuid.uuid4(), uuid.uuid4()
    for _ in range(3):
        await repository.add(_run(workspace_id=workspace_id, sweep_id=mine))
    await repository.add(_run(workspace_id=workspace_id, sweep_id=theirs))
    await repository.add(_run(workspace_id=workspace_id))

    runs = await repository.list_by_sweep(workspace_id, mine)

    assert len(runs) == 3
    assert {run.sweep_id for run in runs} == {mine}


@pytest.mark.asyncio
async def test_list_by_sweep_is_workspace_scoped(sessions) -> None:
    """The filter is in the SQL, not applied after the fetch."""
    repository = SqlAlchemyRunRepository(sessions)
    sweep_id = uuid.uuid4()
    await repository.add(_run(workspace_id=uuid.uuid4(), sweep_id=sweep_id))

    assert await repository.list_by_sweep(uuid.uuid4(), sweep_id) == []


@pytest.mark.asyncio
async def test_sweep_summaries_counts_by_status(sessions) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    workspace_id = uuid.uuid4()
    sweep_id = uuid.uuid4()
    pending = _run(workspace_id=workspace_id, sweep_id=sweep_id)
    finished = _run(workspace_id=workspace_id, sweep_id=sweep_id)
    await repository.add(pending)
    await repository.add(finished)
    finished.start()
    finished.succeed("file:///tmp/x.json")
    await repository.update(finished)

    summaries = await repository.sweep_summaries(workspace_id)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.sweep_id == sweep_id
    assert summary.name == "BBBP comparison"
    assert summary.total == 2
    assert summary.by_status == {"pending": 1, "ready": 1}


@pytest.mark.asyncio
async def test_sweep_summaries_ignores_solo_runs(sessions) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    workspace_id = uuid.uuid4()
    await repository.add(_run(workspace_id=workspace_id))

    assert await repository.sweep_summaries(workspace_id) == []
