"""List a workspace's Datasets, newest first."""

from __future__ import annotations

from dataclasses import dataclass

from returns.result import Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.pagination import (
    PageResult,
    clamp_limit,
    encode_ts_cursor,
    parse_ts_cursor,
)
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.shared.errors import DomainError


@dataclass(frozen=True, kw_only=True)
class ListDatasetsQuery:
    cursor: str | None = None
    limit: int | None = None


class ListDatasets:
    def __init__(self, repository: DatasetRepository) -> None:
        self._repository = repository

    async def __call__(
        self, query: ListDatasetsQuery, auth: AuthContext | None = None
    ) -> Result[PageResult[Dataset], DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        limit = clamp_limit(query.limit)
        # Fetch one more than asked for: if it comes back, there is another page,
        # which is cheaper and more truthful than a COUNT over the whole table.
        datasets = await self._repository.list(
            auth.workspace_id, cursor=parse_ts_cursor(query.cursor), limit=limit + 1
        )
        next_cursor = None
        if len(datasets) > limit:
            datasets = datasets[:limit]
            next_cursor = encode_ts_cursor(datasets[-1].created_at, datasets[-1].id)
        return Success(PageResult(items=datasets, next_cursor=next_cursor))
