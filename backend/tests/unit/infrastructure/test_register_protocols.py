import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest_asyncio
from duar_auth import DuarError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from daikonstudio.infrastructure.duar.protocol_access import RESOURCE_TYPE
from daikonstudio.infrastructure.duar.register_protocols import register_all
from daikonstudio.infrastructure.persistence.sqlalchemy.catalog.repository import (
    SqlAlchemyProtocolRepository,
)
from tests.unit.infrastructure.test_duar_protocol_access import make_protocol


@pytest_asyncio.fixture
async def session_factory(_migrated_engine: AsyncEngine) -> AsyncIterator[async_sessionmaker]:
    async with _migrated_engine.connect() as connection:
        await connection.begin()
        yield async_sessionmaker(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        await connection.rollback()


def _permissions() -> MagicMock:
    permissions = MagicMock()
    permissions.register_resource = AsyncMock(return_value={})
    permissions.update_visibility = AsyncMock(return_value={})
    return permissions


async def _add(session_factory, **kwargs):
    protocol = make_protocol(created_by=kwargs.pop("created_by", uuid.uuid4()))
    if kwargs.get("publish"):
        protocol.publish()
    await SqlAlchemyProtocolRepository(session_factory).add(protocol)
    return protocol


async def test_a_draft_is_registered_private_and_a_published_one_workspace(session_factory):
    draft = await _add(session_factory)
    published = await _add(session_factory, publish=True)
    permissions = _permissions()

    report = await register_all(session_factory, permissions, log=lambda _: None)

    visibility = {
        call.kwargs["resource_id"]: call.kwargs["visibility"]
        for call in permissions.register_resource.await_args_list
    }
    assert visibility == {draft.id: "private", published.id: "workspace"}
    permissions.update_visibility.assert_awaited_once_with(
        "", RESOURCE_TYPE, published.id, "workspace"
    )
    assert report.registered == 2


async def test_a_refusal_is_skipped_and_the_loop_continues(session_factory):
    first = await _add(session_factory)
    second = await _add(session_factory)
    permissions = _permissions()

    async def register(**kwargs):
        if kwargs["resource_id"] == first.id:
            raise DuarError("Owner is not a member", status_code=400)
        return {}

    permissions.register_resource.side_effect = register

    report = await register_all(session_factory, permissions, log=lambda _: None)

    assert report.skipped == 1
    assert report.registered == 1
    assert report.failures == []
    assert permissions.register_resource.await_count == 2
    assert second.id  # both were attempted


async def test_a_server_error_is_a_failure(session_factory):
    await _add(session_factory)
    permissions = _permissions()
    permissions.register_resource.side_effect = DuarError("boom", status_code=502)
    report = await register_all(session_factory, permissions, log=lambda _: None)
    assert len(report.failures) == 1
