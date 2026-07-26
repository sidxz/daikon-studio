"""Persistence port for the InSilicoProtocol aggregate.

Only `add` -- training is the one thing in Phase 1 that *creates* a Protocol,
and this port exists so `TrainProtocol` can persist one without the application
layer importing SQLAlchemy. Reading, listing and publishing arrive with the
routes in Task 16; the port grows the methods those use cases actually call and
not one before.
"""

from typing import Protocol

from daikonstudio.domain.catalog.protocol import InSilicoProtocol


class ProtocolRepository(Protocol):
    async def add(self, protocol: InSilicoProtocol) -> None: ...
