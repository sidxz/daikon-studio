"""SqlAlchemyStudyRepository against real Postgres."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from daikonstudio.domain.shared.errors import ConcurrencyConflictError
from daikonstudio.domain.studies.study import Answered, SlotValue, Step, Study
from daikonstudio.infrastructure.persistence.sqlalchemy.studies.repository import (
    SqlAlchemyStudyRepository,
)


@pytest_asyncio.fixture
async def session_factory(_migrated_engine: AsyncEngine) -> AsyncIterator[async_sessionmaker]:
    async with _migrated_engine.connect() as connection:
        await connection.begin()
        yield async_sessionmaker(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        await connection.rollback()


def _study(workspace_id: uuid.UUID | None = None, created_by: uuid.UUID | None = None) -> Study:
    steps = [
        Step(
            position=i,
            tool_id=t,
            tool_version="1",
            name=t,
            purpose="p",
            params={"pdb": "1ABC"},
            gated=bool(i),
            always=False,
            expanded_from="x" if i else None,
        )
        for i, t in enumerate(("fetch_pdb", "find_pockets"))
    ]
    brought = SlotValue(
        id=uuid.uuid4(),
        key="structure",
        kind="structure",
        blob_key="b/1",
        summary={"atoms": 10},
        produced_by=None,
        asserted_by=uuid.uuid4(),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return Study(
        workspace_id=workspace_id or uuid.uuid4(),
        created_by=created_by or uuid.uuid4(),
        goal="g",
        restated_goal="rg",
        checks=("c1",),
        stop_conditions=("s1", "s2"),
        steps=steps,
        values=[brought],
        answers=[Answered(ask="Which column?", answer="pIC50"), Answered(ask="Why?", answer="")],
        planning={"plan": {"restated_goal": "rg"}},
    )


@pytest.mark.asyncio
async def test_a_study_round_trips_field_for_field(session_factory: async_sessionmaker) -> None:
    repo = SqlAlchemyStudyRepository(session_factory)
    study = _study()
    await repo.add(study)
    got = await repo.get(study.workspace_id, study.id)
    assert got is not None
    assert (got.goal, got.restated_goal, got.checks, got.stop_conditions) == (
        "g",
        "rg",
        ("c1",),
        ("s1", "s2"),
    )
    assert got.created_by == study.created_by and got.version == 1
    assert got.steps == study.steps
    assert got.values == study.values
    assert got.answers == study.answers
    assert got.planning == study.planning
    assert await repo.get(uuid.uuid4(), study.id) is None


@pytest.mark.asyncio
async def test_update_bumps_version_and_a_stale_update_conflicts(
    session_factory: async_sessionmaker,
) -> None:
    repo = SqlAlchemyStudyRepository(session_factory)
    study = _study()
    await repo.add(study)
    stale = await repo.get(study.workspace_id, study.id)
    assert stale is not None
    study.start_step(0)
    await repo.update(study)
    assert study.version == 2
    got = await repo.get(study.workspace_id, study.id)
    assert got is not None and got.version == 2 and got.steps[0].started_at is not None
    stale.start_step(0)
    with pytest.raises(ConcurrencyConflictError):
        await repo.update(stale)


@pytest.mark.asyncio
async def test_list_is_scoped_newest_first_and_filterable(
    session_factory: async_sessionmaker,
) -> None:
    repo = SqlAlchemyStudyRepository(session_factory)
    ws, me = uuid.uuid4(), uuid.uuid4()
    first, second, other = _study(ws, me), _study(ws), _study()
    first.created_at = datetime(2026, 1, 1, tzinfo=UTC)
    second.created_at = datetime(2026, 1, 2, tzinfo=UTC)
    for s in (first, second, other):
        await repo.add(s)
    assert [s.id for s in await repo.list(ws)] == [second.id, first.id]
    assert [s.id for s in await repo.list(ws, created_by=me)] == [first.id]
    assert len(await repo.list(ws, limit=1)) == 1
