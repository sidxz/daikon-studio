import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from duar_auth import DuarError

from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from daikonstudio.domain.shared.errors import ServiceUnavailableError
from daikonstudio.infrastructure.duar.protocol_access import RESOURCE_TYPE, DuarProtocolAccess


def make_protocol(*, created_by: uuid.UUID | None) -> InSilicoProtocol:
    return InSilicoProtocol(
        workspace_id=uuid.uuid4(),
        name="p",
        dataset_id=uuid.uuid4(),
        engine_id="ecfp4-xgboost",
        artifact_uri="file:///x",
        readouts=(),
        conditions={},
        created_by=created_by,
    )


def _duar() -> MagicMock:
    duar = MagicMock()
    duar.permissions.register_resource = AsyncMock(return_value={})
    duar.permissions.deregister_resource = AsyncMock(return_value=None)
    duar.permissions.update_visibility = AsyncMock(return_value={})
    return duar


async def test_register_makes_a_draft_private_owned_by_its_creator():
    duar = _duar()
    protocol = make_protocol(created_by=uuid.uuid4())
    await DuarProtocolAccess(duar, retry_delays=()).register(protocol)
    duar.permissions.register_resource.assert_awaited_once_with(
        resource_type=RESOURCE_TYPE,
        resource_id=protocol.id,
        workspace_id=protocol.workspace_id,
        owner_id=protocol.created_by,
        visibility="private",
    )


async def test_register_retries_a_transient_failure_then_gives_up_without_raising():
    duar = _duar()
    duar.permissions.register_resource.side_effect = httpx.ConnectError("down")
    await DuarProtocolAccess(duar, retry_delays=(0, 0)).register(
        make_protocol(created_by=uuid.uuid4())
    )
    assert duar.permissions.register_resource.await_count == 3


async def test_register_does_not_retry_a_refusal():
    duar = _duar()
    duar.permissions.register_resource.side_effect = DuarError(
        "Owner is not a member", status_code=400
    )
    await DuarProtocolAccess(duar, retry_delays=(0, 0)).register(
        make_protocol(created_by=uuid.uuid4())
    )
    assert duar.permissions.register_resource.await_count == 1


async def test_register_skips_a_protocol_with_no_creator():
    duar = _duar()
    await DuarProtocolAccess(duar).register(make_protocol(created_by=None))
    duar.permissions.register_resource.assert_not_awaited()


async def test_deregister_tolerates_an_unknown_resource():
    duar = _duar()
    duar.permissions.deregister_resource.side_effect = DuarError("not found", status_code=404)
    await DuarProtocolAccess(duar).deregister(uuid.uuid4())  # no raise


async def test_visible_ids_is_none_on_full_access():
    auth = MagicMock(accessible=AsyncMock(return_value=([], True)))
    assert await DuarProtocolAccess(_duar()).visible_ids(auth) is None
    auth.accessible.assert_awaited_once_with(RESOURCE_TYPE, "view")


async def test_visible_ids_returns_the_set():
    ids = [uuid.uuid4(), uuid.uuid4()]
    auth = MagicMock(accessible=AsyncMock(return_value=(ids, False)))
    assert await DuarProtocolAccess(_duar()).visible_ids(auth) == frozenset(ids)


async def test_reads_fail_closed_when_duar_is_down():
    auth = MagicMock(
        accessible=AsyncMock(side_effect=httpx.ConnectError("down")),
        can=AsyncMock(side_effect=DuarError("boom", status_code=502)),
    )
    access = DuarProtocolAccess(_duar())
    with pytest.raises(ServiceUnavailableError):
        await access.visible_ids(auth)
    with pytest.raises(ServiceUnavailableError):
        await access.can_view(auth, uuid.uuid4())


async def test_publish_registers_as_workspace_when_duar_never_heard_of_it():
    duar = _duar()
    protocol = make_protocol(created_by=uuid.uuid4())
    auth = MagicMock(update_visibility=AsyncMock(side_effect=DuarError("nf", status_code=404)))
    await DuarProtocolAccess(duar).make_workspace_visible(auth, protocol)
    duar.permissions.register_resource.assert_awaited_once()
    assert duar.permissions.register_resource.await_args.kwargs["visibility"] == "workspace"


async def test_publish_flips_visibility():
    protocol = make_protocol(created_by=uuid.uuid4())
    auth = MagicMock(update_visibility=AsyncMock(return_value={}))
    await DuarProtocolAccess(_duar()).make_workspace_visible(auth, protocol)
    auth.update_visibility.assert_awaited_once_with(RESOURCE_TYPE, protocol.id, "workspace")
