"""List a workspace's Runs, newest first.

The API exposes `POST /runs/{id}/cancel`, which only means anything if a Run
outlives the page that started it -- and until this existed, nothing could
find one again. A prediction over a large compound set is exactly the case
where a scientist closes the tab and comes back.

`kind` is optional because the two kinds are reached differently: a training
Run belongs to its Protocol's history, while prediction Runs are the list a
user browses. Filtering server-side keeps a client from paging through
thousands of the wrong kind to assemble one screen.

`protocol_id` is the other half of that sentence, and it went missing until
now: migration 007 created `ix_runs_workspace_protocol_id` and documented it as
backing "which runs belong to this Protocol -- the Protocol detail page's run
history, and the only query this column exists to serve", and no query used it.
It does now.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.pagination import (
    PageResult,
    clamp_limit,
    encode_ts_cursor,
    parse_ts_cursor,
)
from daikonstudio.application.ports.protocol_access import ProtocolAccess
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.run_repository import RunRepository, TrainingVisibility
from daikonstudio.domain.execution.run import Run, RunKind, RunStatus
from daikonstudio.domain.shared.errors import DomainError, ValidationError


@dataclass(frozen=True, kw_only=True)
class ListRunsQuery:
    kind: RunKind | None = None
    protocol_id: uuid.UUID | None = None
    cursor: str | None = None
    limit: int | None = None
    # Only the runs the caller started.
    mine: bool = False
    statuses: tuple[RunStatus, ...] = ()
    # Only runs of the protocols filed in this folder.
    folder_id: uuid.UUID | None = None
    # Runs whose name contains this text, case-insensitively.
    q: str | None = None


class ListRuns:
    def __init__(
        self, repository: RunRepository, access: ProtocolAccess, protocols: ProtocolRepository
    ) -> None:
        self._access = access
        self._repository = repository
        self._protocols = protocols

    async def __call__(
        self, query: ListRunsQuery, auth: AuthContext | None = None
    ) -> Result[PageResult[Run], DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        limit = clamp_limit(query.limit)
        try:
            cursor = parse_ts_cursor(query.cursor)
        except ValidationError as error:
            return Failure(error)
        # One more than asked for: if it comes back there is another page, which
        # is cheaper and more truthful than a COUNT over the whole table.
        visible = await self._access.visible_ids(auth)
        training_visible_to = (
            None
            if visible is None
            else TrainingVisibility(user_id=auth.user_id, protocol_ids=visible)
        )
        protocol_ids = None
        if query.folder_id is not None:
            protocol_ids = frozenset(
                await self._protocols.ids_in_folder(auth.workspace_id, query.folder_id)
            )
            if query.protocol_id is not None:
                protocol_ids &= {query.protocol_id}
        runs = await self._repository.list(
            auth.workspace_id,
            kind=query.kind,
            protocol_id=query.protocol_id,
            protocol_ids=protocol_ids,
            requested_by=auth.user_id if query.mine else None,
            statuses=query.statuses,
            name_contains=(query.q or "").strip() or None,
            cursor=cursor,
            limit=limit + 1,
            training_visible_to=training_visible_to,
        )
        next_cursor = None
        if len(runs) > limit:
            runs = runs[:limit]
            next_cursor = encode_ts_cursor(runs[-1].created_at, runs[-1].id)
        return Success(PageResult(items=runs, next_cursor=next_cursor))
