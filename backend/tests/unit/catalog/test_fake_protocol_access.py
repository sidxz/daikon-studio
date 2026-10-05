import uuid

from tests.fakes.auth import FakeAuth
from tests.fakes.protocol_access import FakeProtocolAccess
from tests.unit.infrastructure.test_duar_protocol_access import make_protocol


async def _setup():
    owner = FakeAuth()
    protocol = make_protocol(created_by=owner.user_id)
    protocol.workspace_id = owner.workspace_id
    access = FakeProtocolAccess()
    await access.register(protocol)
    return access, protocol, owner


async def test_the_owner_sees_their_draft():
    access, protocol, owner = await _setup()
    assert await access.visible_ids(owner) == frozenset({protocol.id})
    assert await access.can_view(owner, protocol.id)


async def test_another_member_does_not():
    access, protocol, owner = await _setup()
    other = FakeAuth(workspace_id=owner.workspace_id, user_id=uuid.uuid4())
    assert await access.visible_ids(other) == frozenset()
    assert not await access.can_view(other, protocol.id)


async def test_an_admin_does():
    access, protocol, owner = await _setup()
    admin = FakeAuth(workspace_id=owner.workspace_id, workspace_role="admin")
    assert await access.visible_ids(admin) is None
    assert await access.can_view(admin, protocol.id)


async def test_after_publishing_the_other_member_does():
    access, protocol, owner = await _setup()
    other = FakeAuth(workspace_id=owner.workspace_id, user_id=uuid.uuid4())
    await access.make_workspace_visible(owner, protocol)
    assert await access.can_view(other, protocol.id)
