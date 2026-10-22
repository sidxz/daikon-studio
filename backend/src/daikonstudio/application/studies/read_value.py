"""Reading one value a study holds, as much of it as a page shows."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

import polars as pl
from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.study_repository import StudyRepository
from daikonstudio.application.studies.get_study import GetStudy
from daikonstudio.application.tools import codecs
from daikonstudio.domain.shared.errors import DomainError, NotFoundError
from daikonstudio.domain.studies.study import SlotValue


@dataclass(frozen=True)
class ValueView:
    value: SlotValue
    data: Any


class ReadValue:
    def __init__(self, studies: StudyRepository, store: BlobStore) -> None:
        self._get = GetStudy(studies)
        self._store = store

    async def __call__(
        self,
        study_id: uuid.UUID,
        value_id: uuid.UUID,
        auth: AuthContext | None = None,
        *,
        limit: int = 200,
    ) -> Result[ValueView, DomainError]:
        found = await self._get(study_id, auth)
        if not isinstance(found, Success):
            return found  # type: ignore[return-value]
        value = next((v for v in found.unwrap().values if v.id == value_id), None)
        if value is None:
            return Failure(NotFoundError("Value", str(value_id)))
        decoded = codecs.decode(
            value.kind, await asyncio.to_thread(self._store.get_bytes, value.blob_key)
        )
        # ponytail: 200-row preview; page it when a table is read for more than a look.
        data: Any
        if isinstance(decoded, pl.DataFrame):
            # fill_nan: a NaN is not JSON, and a float column can carry one.
            data = decoded.head(limit).fill_nan(None).to_dicts()
        elif isinstance(decoded, dict):
            data = {k: v for k, v in decoded.items() if k != "text"}
        elif isinstance(decoded, list):
            data = decoded[:limit]
        else:
            data = decoded
        return Success(ValueView(value, data))
