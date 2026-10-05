"""Read one or many Protocols, scoped to the caller's workspace.

Drafts are included in both `ListProtocols` and `GetProtocol` -- a workspace's
own drafts are exactly the models it just trained and has not yet decided
whether to publish, and hiding them would make `POST /protocols` invisible
until someone published its result. `status` (`"draft"`/`"published"`, plus
the derived `is_locked`) is how a caller tells the two apart; nothing about a
draft is hidden from its own workspace, only unpublishable and (per Task 12's
`InSilicoProtocol.publish` docstring) not yet the thing later tasks may treat
as citable or runnable by anyone outside it.

`GetProtocol` lives here rather than in its own file: it is the same shape as
`ListProtocols` (one read, one repository, no other collaborator) and Task 16
does not otherwise need a fourth application/catalog module for it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.catalog.visibility import visible_protocol
from daikonstudio.application.pagination import (
    PageResult,
    clamp_limit,
    encode_ts_cursor,
    parse_ts_cursor,
)
from daikonstudio.application.ports.protocol_access import ProtocolAccess
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from daikonstudio.domain.shared.errors import DomainError, NotFoundError, ValidationError


@dataclass(frozen=True, kw_only=True)
class ListProtocolsQuery:
    cursor: str | None = None
    limit: int | None = None
    # Only the Protocols trained on this Dataset: what stands between it and deletion.
    dataset_id: uuid.UUID | None = None
    # Only the Protocols the caller created.
    mine: bool = False


class ListProtocols:
    def __init__(self, repository: ProtocolRepository, access: ProtocolAccess) -> None:
        self._repository = repository
        self._access = access

    async def __call__(
        self, query: ListProtocolsQuery, auth: AuthContext | None = None
    ) -> Result[PageResult[InSilicoProtocol], DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        limit = clamp_limit(query.limit)
        try:
            cursor = parse_ts_cursor(query.cursor)
        except ValidationError as error:
            return Failure(error)
        # Fetch one more than asked for: if it comes back, there is another page,
        # which is cheaper and more truthful than a COUNT over the whole table.
        visible = await self._access.visible_ids(auth)
        protocols = await self._repository.list(
            auth.workspace_id,
            cursor=cursor,
            limit=limit + 1,
            dataset_id=query.dataset_id,
            only_ids=visible,
            created_by=auth.user_id if query.mine else None,
        )
        next_cursor = None
        if len(protocols) > limit:
            protocols = protocols[:limit]
            next_cursor = encode_ts_cursor(protocols[-1].created_at, protocols[-1].id)
        return Success(PageResult(items=protocols, next_cursor=next_cursor))


@dataclass(frozen=True, kw_only=True)
class GetProtocolQuery:
    protocol_id: uuid.UUID


class GetProtocol:
    def __init__(self, repository: ProtocolRepository, access: ProtocolAccess) -> None:
        self._repository = repository
        self._access = access

    async def __call__(
        self, query: GetProtocolQuery, auth: AuthContext | None = None
    ) -> Result[InSilicoProtocol, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        protocol = await visible_protocol(
            self._repository, self._access, auth, auth.workspace_id, query.protocol_id
        )
        if protocol is None:
            return Failure(NotFoundError("Protocol", str(query.protocol_id)))
        return Success(protocol)
