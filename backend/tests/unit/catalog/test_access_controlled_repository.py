import uuid

from daikonstudio.application.catalog.access_controlled_repository import (
    AccessControlledProtocolRepository,
)
from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from tests.fakes.protocol_access import FakeProtocolAccess
from tests.unit.infrastructure.test_duar_protocol_access import make_protocol


class _Inner:
    def __init__(self, calls: list[str]) -> None:
        self.rows: dict[uuid.UUID, InSilicoProtocol] = {}
        self.calls = calls

    async def add(self, protocol: InSilicoProtocol) -> None:
        self.calls.append("inner.add")
        self.rows[protocol.id] = protocol

    async def update(self, protocol: InSilicoProtocol) -> None:
        self.rows[protocol.id] = protocol

    async def delete(self, workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> None:
        self.calls.append("inner.delete")
        self.rows.pop(protocol_id, None)

    async def get(self, workspace_id, protocol_id):  # type: ignore[no-untyped-def]
        return self.rows.get(protocol_id)

    async def list(self, workspace_id, **_):  # type: ignore[no-untyped-def]
        return list(self.rows.values())


class _Access(FakeProtocolAccess):
    calls: list[str]

    async def register(self, protocol: InSilicoProtocol) -> None:
        self.calls.append("access.register")
        await super().register(protocol)

    async def deregister(self, protocol_id: uuid.UUID) -> None:
        self.calls.append("access.deregister")
        await super().deregister(protocol_id)


def _build() -> tuple[AccessControlledProtocolRepository, _Inner, _Access, list[str]]:
    calls: list[str] = []
    inner, access = _Inner(calls), _Access()
    access.calls = calls
    return AccessControlledProtocolRepository(inner, access), inner, access, calls  # type: ignore[arg-type]


async def test_add_registers_before_inserting():
    repo, inner, access, calls = _build()
    protocol = make_protocol(created_by=uuid.uuid4())
    await repo.add(protocol)
    assert calls == ["access.register", "inner.add"]
    assert protocol.id in access.acl and protocol.id in inner.rows


async def test_a_protocol_whose_registration_was_skipped_is_still_added():
    # The real access logs and returns on failure; a creator-less protocol takes that path.
    repo, inner, access, _ = _build()
    protocol = make_protocol(created_by=None)
    await repo.add(protocol)
    assert protocol.id in inner.rows and protocol.id not in access.acl


async def test_delete_removes_the_row_then_deregisters():
    repo, inner, access, calls = _build()
    protocol = make_protocol(created_by=uuid.uuid4())
    await repo.add(protocol)
    calls.clear()
    await repo.delete(protocol.workspace_id, protocol.id)
    assert calls == ["inner.delete", "access.deregister"]
    assert protocol.id not in inner.rows and protocol.id not in access.acl


async def test_reads_and_updates_delegate_unchanged():
    repo, _, _, calls = _build()
    protocol = make_protocol(created_by=uuid.uuid4())
    await repo.add(protocol)
    calls.clear()
    assert await repo.get(protocol.workspace_id, protocol.id) is protocol
    assert await repo.list(protocol.workspace_id) == [protocol]
    await repo.update(protocol)
    assert calls == []
