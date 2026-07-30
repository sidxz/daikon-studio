"""List a workspace's Collections, newest first.

A Collection is this product's deliverable -- the curated set of compounds
someone is willing to order -- and until this existed there was no way to
reach one again after the tab that created it was closed. Every other
Phase 1 artifact is a means; this is the end, and it was the only one with
no way back to it.
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
from daikonstudio.application.ports.collection_repository import CollectionRepository
from daikonstudio.domain.data.collection import Collection
from daikonstudio.domain.shared.errors import DomainError, ValidationError


@dataclass(frozen=True, kw_only=True)
class ListCollectionsQuery:
    cursor: str | None = None
    limit: int | None = None


class ListCollections:
    def __init__(self, repository: CollectionRepository) -> None:
        self._repository = repository

    async def __call__(
        self, query: ListCollectionsQuery, auth: AuthContext | None = None
    ) -> Result[PageResult[Collection], DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        limit = clamp_limit(query.limit)
        try:
            cursor = parse_ts_cursor(query.cursor)
        except ValidationError as error:
            return Failure(error)
        # One more than asked for: if it comes back there is another page, which
        # is cheaper and more truthful than a COUNT over the whole table.
        collections = await self._repository.list(
            auth.workspace_id, cursor=cursor, limit=limit + 1
        )
        next_cursor = None
        if len(collections) > limit:
            collections = collections[:limit]
            next_cursor = encode_ts_cursor(collections[-1].created_at, collections[-1].id)
        return Success(PageResult(items=collections, next_cursor=next_cursor))
