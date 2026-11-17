"""Reading studies back. A study is its author's: anyone else gets a 404."""

from __future__ import annotations

import uuid

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.ports.study_repository import StudyRepository
from daikonstudio.domain.shared.errors import DomainError, NotFoundError
from daikonstudio.domain.studies.study import Study


class GetStudy:
    def __init__(self, studies: StudyRepository) -> None:
        self._studies = studies

    async def __call__(
        self, study_id: uuid.UUID, auth: AuthContext | None = None
    ) -> Result[Study, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        study = await self._studies.get(auth.workspace_id, study_id)
        if study is None or study.created_by != auth.user_id:
            return Failure(NotFoundError("Study", str(study_id)))
        return Success(study)


class ListStudies:
    def __init__(self, studies: StudyRepository) -> None:
        self._studies = studies

    async def __call__(self, auth: AuthContext | None = None) -> Result[list[Study], DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        return Success(await self._studies.list(auth.workspace_id, created_by=auth.user_id))
