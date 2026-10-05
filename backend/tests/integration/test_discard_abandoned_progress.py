"""Saved progress of training runs stopped long ago is deleted; nothing else is."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from daikonstudio.application.engines.checkpoints import checkpoint_root
from daikonstudio.application.execution.discard_abandoned_progress import (
    SAVED_PROGRESS_DAYS,
    DiscardAbandonedProgress,
)
from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.repository import (
    SqlAlchemyRunRepository,
)
from tests.fakes.blob_store import InMemoryBlobStore


@pytest_asyncio.fixture
async def session_factory(_migrated_engine: AsyncEngine) -> AsyncIterator[async_sessionmaker]:
    async with _migrated_engine.connect() as connection:
        await connection.begin()
        yield async_sessionmaker(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        await connection.rollback()


async def test_only_long_stopped_training_runs_lose_their_saved_progress(session_factory):
    runs = SqlAlchemyRunRepository(session_factory)
    store = InMemoryBlobStore()
    workspace, dataset = uuid.uuid4(), uuid.uuid4()

    async def training(stop: str | None) -> str:
        run = Run(
            kind=RunKind.TRAINING,
            workspace_id=workspace,
            requested_by=uuid.uuid4(),
            cache_key="deadbeef",
            params={"dataset_id": str(dataset)},
        )
        await runs.add(run)
        run.start()
        if stop == "cancel":
            run.cancel()
        elif stop == "fail":
            run.fail("out of memory")
        await runs.update(run)
        saved = checkpoint_root(workspace, dataset, run.id) + "training-state.a"
        store.put_bytes(saved, b"weights")
        return saved

    cancelled, failed, running = (
        await training("cancel"),
        await training("fail"),
        await training(None),
    )
    discard = DiscardAbandonedProgress(runs, store)

    await discard()  # stopped just now: Resume still has its progress
    assert all(store.exists(key) for key in (cancelled, failed, running))

    await discard(now=datetime.now(UTC) + timedelta(days=SAVED_PROGRESS_DAYS, hours=1))
    assert not store.exists(cancelled)
    assert not store.exists(failed)
    assert store.exists(running)  # a run still training keeps its progress at any age
