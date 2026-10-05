"""A shared folder for datasets or for protocols. Shared across domain contexts, so it
lives here and none of `catalog`, `data` or `execution` has to import another."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from daikonstudio.domain.shared.errors import ValidationError

MAX_FOLDER_NAME = 100


class FolderKind(StrEnum):
    DATASET = "dataset"
    PROTOCOL = "protocol"


@dataclass(frozen=True, kw_only=True)
class Folder:
    workspace_id: uuid.UUID
    kind: FolderKind
    name: str
    created_by: uuid.UUID
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def clean_folder_name(raw: str) -> str:
    name = raw.strip()
    if not 1 <= len(name) <= MAX_FOLDER_NAME:
        raise ValidationError("A folder name must be 1 to 100 characters.")
    return name
