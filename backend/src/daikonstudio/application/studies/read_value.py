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


def decode_view(kind: str, blob: bytes, *, limit: int | None = 200) -> Any:
    """A value as a page shows it, and as the agent reads it: JSON-safe, without the file text.

    `limit` caps a table's rows or a list's items; `None` means all of them, for a check.
    """
    decoded = codecs.decode(kind, blob)
    # ponytail: 200-row preview; page it when a table is read for more than a look.
    if isinstance(decoded, pl.DataFrame):
        frame = decoded if limit is None else decoded.head(limit)
        # fill_nan: a NaN is not JSON, and a float column can carry one.
        return frame.fill_nan(None).to_dicts()
    if isinstance(decoded, dict):
        return {k: v for k, v in decoded.items() if k != "text"}
    if isinstance(decoded, list):
        return decoded[:limit]
    return decoded


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
        blob = await asyncio.to_thread(self._store.get_bytes, value.blob_key)
        return Success(ValueView(value, decode_view(value.kind, blob, limit=limit)))
