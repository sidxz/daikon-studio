"""Read one Dataset, scoped to the caller's workspace.

A Dataset belonging to another workspace comes back as 404, never 403: a 403
would confirm that the id exists, which is exactly what a caller probing for
another tenant's data wants to learn.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.shared.errors import DomainError, NotFoundError


@dataclass(frozen=True, kw_only=True)
class GetDatasetQuery:
    dataset_id: uuid.UUID


class GetDataset:
    def __init__(self, repository: DatasetRepository) -> None:
        self._repository = repository

    async def __call__(
        self, query: GetDatasetQuery, auth: AuthContext | None = None
    ) -> Result[Dataset, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        dataset = await self._repository.get(auth.workspace_id, query.dataset_id)
        if dataset is None:
            return Failure(NotFoundError("Dataset", str(query.dataset_id)))
        return Success(dataset)
