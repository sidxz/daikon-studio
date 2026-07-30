"""List a workspace's Runs, newest first.

The API exposes `POST /runs/{id}/cancel`, which only means anything if a Run
outlives the page that started it -- and until this existed, nothing could
find one again. A prediction over a large compound set is exactly the case
where a scientist closes the tab and comes back.

`kind` is optional because the two kinds are reached differently: a training
Run belongs to its Protocol's history, while prediction Runs are the list a
user browses. Filtering server-side keeps a client from paging through
thousands of the wrong kind to assemble one screen.
"""

from __future__ import annotations

from dataclasses import dataclass

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.pagination import (
    PageResult,
    clamp_limit,
    encode_ts_cursor,
    parse_ts_cursor,
)
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.domain.shared.errors import DomainError, ValidationError


@dataclass(frozen=True, kw_only=True)
class ListRunsQuery:
    kind: RunKind | None = None
    cursor: str | None = None
    limit: int | None = None


class ListRuns:
    def __init__(self, repository: RunRepository) -> None:
        self._repository = repository

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
        runs = await self._repository.list(
            auth.workspace_id, kind=query.kind, cursor=cursor, limit=limit + 1
        )
        next_cursor = None
        if len(runs) > limit:
            runs = runs[:limit]
            next_cursor = encode_ts_cursor(runs[-1].created_at, runs[-1].id)
        return Success(PageResult(items=runs, next_cursor=next_cursor))
