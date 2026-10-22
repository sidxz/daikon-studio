"""SqlAlchemyStudyRepository against real Postgres."""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from daikonstudio.domain.shared.errors import ConcurrencyConflictError
from daikonstudio.domain.studies.study import (
    Acted,
    Answered,
    Asked,
    Pointer,
    Proposed,
    SlotValue,
    Step,
    Study,
    Thread,
    Turn,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.studies.repository import (
    SqlAlchemyStudyRepository,
)

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


@pytest_asyncio.fixture
async def session_factory(_migrated_engine: AsyncEngine) -> AsyncIterator[async_sessionmaker]:
    async with _migrated_engine.connect() as connection:
        await connection.begin()
        yield async_sessionmaker(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        await connection.rollback()


def _value(*, produced_by: uuid.UUID | None, **kw: Any) -> SlotValue:
    return SlotValue(
        id=uuid.uuid4(),
        key="structure",
        kind="structure",
        blob_key=f"b/{uuid.uuid4()}",
        summary={"atoms": 10},
        produced_by=produced_by,
        created_at=_T0,
        **kw,
    )


def _study(
    workspace_id: uuid.UUID | None = None, created_by: uuid.UUID | None = None, **kw: Any
) -> Study:
    steps = [
        Step(
            id=uuid.uuid4(),
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
    return Study(
        workspace_id=workspace_id or uuid.uuid4(),
        created_by=created_by or uuid.uuid4(),
        goal="g",
        restated_goal="rg",
        checks=("c1",),
        stop_conditions=("s1", "s2"),
        steps=steps,
        # One the scientist brought, one a step made: produced_by is null and a step id.
        values=[
            _value(produced_by=None, asserted_by=uuid.uuid4()),
            _value(produced_by=steps[0].id),
        ],
        answers=[Answered(ask="Which column?", answer="pIC50"), Answered(ask="Why?", answer="")],
        **kw,
    )


@pytest.mark.asyncio
async def test_a_study_round_trips_field_for_field(session_factory: async_sessionmaker) -> None:
    repo = SqlAlchemyStudyRepository(session_factory)
    study = _study(mode="walk")  # not the default, so a mapping that drops the column shows
    study.steps[1].problem = "There is no structure to look for pockets in."
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
    assert (got.started_at, got.mode, got.amended_after_steps) == (None, "walk", None)
    assert got.thread == Thread()
    assert await repo.get(uuid.uuid4(), study.id) is None


@pytest.mark.asyncio
async def test_a_draft_round_trips_its_thread_and_started_at(
    session_factory: async_sessionmaker,
) -> None:
    repo = SqlAlchemyStudyRepository(session_factory)
    study = _study()
    fetch, pockets = study.steps
    # Every field of every turn is set to something other than its default, and the
    # nullable ones are set both ways, so a field the mapping drops shows as a difference.
    study.say(Turn(role="scientist", say="Dock into the 1ABC pocket.", about=fetch.id, at=_T0))
    study.say(
        Turn(
            role="agent",
            kind="planning",
            say="Which pocket, and what counts as a hit?",
            questions=(
                Asked(
                    ask="Which pocket?", why="There are three.", options=("A", "B"), default="A"
                ),
                Asked(ask="What counts as a hit?", why="It sets the cutoff."),
            ),
            points=(
                Pointer(
                    step=pockets.id,
                    at="pocket 2",
                    label="the cleft",
                    evidence="the deepest",
                    cites=("score", "volume"),
                ),
                Pointer(step=None, at="chain A", label="the chain", evidence="the only one"),
            ),
            proposals=(
                Proposed(
                    id=uuid.uuid4(),
                    action={
                        "does": "swap",
                        "step": str(pockets.id),
                        "tool_id": "find_pockets_v2",
                        "settings": {"top": 3, "ranked": True},
                    },
                    why="The newer tool ranks by depth.",
                ),
                Proposed(
                    id=uuid.uuid4(),
                    action={"does": "remove", "step": str(fetch.id)},
                    why="The structure is already attached.",
                    applied_at=_T0 + timedelta(minutes=3),
                ),
            ),
            unresolved=("Which ligand?",),
            actions=(
                Acted(
                    does="swap",
                    step=pockets.id,
                    detail="find_pockets for find_pockets_v2",
                    quoting="use the newer one",
                ),
                Acted(does="note", step=None, detail="the goal", quoting="dock it"),
            ),
            looked=("fetch_pdb", "find_pockets"),
            dropped=2,
            silent=True,
            written_by="claude-fable-5-1",
            at=_T0 + timedelta(minutes=1),
        )
    )
    study.thread.summary = "Docking into the 1ABC pocket; which pocket is still to be chosen."
    await repo.add(study)

    got = await repo.get(study.workspace_id, study.id)
    assert got is not None
    assert (got.started_at, got.amended_after_steps) == (None, None)
    assert got.thread.summary == study.thread.summary
    for turn, want in zip(got.thread.turns, study.thread.turns, strict=True):
        for field in dataclasses.fields(Turn):
            assert getattr(turn, field.name) == getattr(want, field.name), field.name


@pytest.mark.asyncio
async def test_update_persists_terms_and_started_at(session_factory: async_sessionmaker) -> None:
    repo = SqlAlchemyStudyRepository(session_factory)
    study = _study()
    await repo.add(study)
    first = study.steps[0]

    # Every column `update` writes, changed the way the services change it.
    study.restated_goal = "rg, once the questions were answered"
    study.answers = (Answered(ask="Which column?", answer="pKi"),)
    study.say(Turn(role="scientist", say="Go ahead.", at=_T0))
    study.start()
    study.start_step(first.id)
    study.finish_step(first.id, [_value(produced_by=first.id)])
    study.amend_terms(("c1", "c2"), ("s1",))
    await repo.update(study)
    assert study.version == 2

    got = await repo.get(study.workspace_id, study.id)
    assert got is not None
    assert (got.restated_goal, got.checks, got.stop_conditions) == (
        "rg, once the questions were answered",
        ("c1", "c2"),
        ("s1",),
    )
    assert got.answers == (Answered(ask="Which column?", answer="pKi"),)
    assert got.started_at is not None and got.started_at == study.started_at
    assert got.amended_after_steps == 1  # one step had run when the terms changed
    assert got.steps == study.steps and got.steps[0].finished_at is not None
    assert got.values == study.values
    assert got.thread == study.thread
    assert (got.updated_at, got.version) == (study.updated_at, 2)


@pytest.mark.asyncio
async def test_update_bumps_version_and_a_stale_update_conflicts(
    session_factory: async_sessionmaker,
) -> None:
    repo = SqlAlchemyStudyRepository(session_factory)
    study = _study()
    await repo.add(study)
    stale = await repo.get(study.workspace_id, study.id)
    assert stale is not None
    study.start_step(study.steps[0].id)
    await repo.update(study)
    assert study.version == 2
    got = await repo.get(study.workspace_id, study.id)
    assert got is not None and got.version == 2 and got.steps[0].started_at is not None
    stale.start_step(stale.steps[0].id)
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
