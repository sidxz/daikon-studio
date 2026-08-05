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
