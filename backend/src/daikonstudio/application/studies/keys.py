"""Where a study's values live in the blob store."""

from __future__ import annotations

import uuid


def value_key(
    workspace_id: uuid.UUID, study_id: uuid.UUID, value_id: uuid.UUID, extension: str
) -> str:
    return f"{workspace_id}/studies/{study_id}/values/{value_id}.{extension}"
