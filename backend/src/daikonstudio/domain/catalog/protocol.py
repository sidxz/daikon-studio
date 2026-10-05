"""The in-silico Protocol aggregate: a trained, published, runnable model.

Publishing is irreversible. `publish()` locks the aggregate; nothing about a
published Protocol's readouts, artifact or conditions changes afterward --
that immutability is what makes a Protocol citable in a paper's methods
section, the same way a Dataset's content hash makes it citable evidence.

Versioning is a self-referencing chain (`parent_protocol_id` +
`protocol_version`), exactly as the sibling assay-protocol project does for
lab protocols. There is deliberately no separate version entity: `new_version`
just returns a fresh DRAFT pointing back at the protocol it superseded.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from daikonstudio.domain.catalog.readout import Readout
from daikonstudio.domain.shared.entity import AggregateRoot
from daikonstudio.domain.shared.errors import ConflictError, DataLockedError


class ProtocolStatus(StrEnum):
    DRAFT = "draft"
    PUBLISHED = "published"


class InSilicoProtocol(AggregateRoot):
    def __init__(
        self,
        *,
        workspace_id: uuid.UUID,
        name: str,
        dataset_id: uuid.UUID,
        engine_id: str,
        artifact_uri: str,
        readouts: tuple[Readout, ...],
        conditions: Mapping[str, Any],
        status: ProtocolStatus = ProtocolStatus.DRAFT,
        published_at: datetime | None = None,
        parent_protocol_id: uuid.UUID | None = None,
        protocol_version: int = 1,
        created_by: uuid.UUID | None = None,
        folder_id: uuid.UUID | None = None,
        id: uuid.UUID | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        version: int = 1,
    ) -> None:
        super().__init__(id=id, created_at=created_at, updated_at=updated_at, version=version)
        self.workspace_id = workspace_id
        self.name = name
        self.dataset_id = dataset_id
        self.engine_id = engine_id
        self.artifact_uri = artifact_uri
        self.readouts = readouts
        # Copied, not aliased: `readouts` is safe to hold by reference because it's a
        # tuple of frozen dataclasses, but a raw dict is not. Without this copy, either
        # the caller mutating their own dict after construction, or `new_version()`
        # handing this same dict to a child, would silently reach back into an
        # already-published, supposedly-immutable Protocol. Stored privately and
        # exposed only through the `conditions` property's read-only view: a copy
        # alone stops aliasing between two *objects*, but a caller holding this
        # object itself could still write straight through a plain dict attribute.
        self._conditions: dict[str, Any] = dict(conditions)
        self.status = status
        self.published_at = published_at
        self.parent_protocol_id = parent_protocol_id
        self.protocol_version = protocol_version
        # Who trained it, for the delete permission (backfilled by migration 011
        # from the training run's `requested_by`).
        self.created_by = created_by
        # The shared folder it is filed in, if any. Organisation only.
        self.folder_id = folder_id

    @property
    def conditions(self) -> Mapping[str, Any]:
        """A read-only view, not a copy: an in-place write (`protocol.conditions["x"]
        = 1`) raises `TypeError` at the point of the mistake. A copy-on-read property
        would let that same write appear to succeed and then silently vanish, which
        is a worse failure mode than a loud one for something citable."""
        return MappingProxyType(self._conditions)

    @property
    def is_locked(self) -> bool:
        return self.status is ProtocolStatus.PUBLISHED

    def publish(self) -> None:
        """Set status to PUBLISHED and stamp `published_at`. Idempotent this is not:
        a second call raises `DataLockedError` (mapped to 423) rather than silently
        no-op'ing, because a caller retrying a publish should learn the protocol is
        already locked, not be quietly told it succeeded again."""
        if self.status is ProtocolStatus.PUBLISHED:
            raise DataLockedError("This protocol is already published.")
        self.status = ProtocolStatus.PUBLISHED
        self.published_at = datetime.now(UTC)
        self.updated_at = self.published_at

    def new_version(self, *, artifact_uri: str) -> InSilicoProtocol:
        """A fresh DRAFT chained to this protocol, carrying forward everything
        that describes what it predicts (readouts, conditions, dataset, engine).
        Only `artifact_uri` changes going in -- retraining produced new weights,
        not a new contract.

        Only a published Protocol has anything meaningful to version *from*: the
        aggregate has no draft-editing method, so a chain that could fork off an
        unpublished draft would let an unbounded pile of never-published "versions"
        accumulate for no reason. Want to change a draft before it's published?
        Make a new Protocol -- that's the only edit a draft supports anyway.
        """
        if not self.is_locked:
            raise ConflictError(
                f"Protocol '{self.id}' has not been published; there is nothing to version from"
            )
        return InSilicoProtocol(
            workspace_id=self.workspace_id,
            name=self.name,
            dataset_id=self.dataset_id,
            engine_id=self.engine_id,
            artifact_uri=artifact_uri,
            readouts=self.readouts,
            conditions=self.conditions,
            parent_protocol_id=self.id,
            protocol_version=self.protocol_version + 1,
        )
