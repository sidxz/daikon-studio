"""Keyset-cursor helpers shared by paginated queries.

Keyset, not OFFSET: a listing sorted newest-first is exactly where rows get
inserted, and OFFSET pagination silently repeats or skips rows when that happens
mid-scroll. The cursor is opaque to clients but is really `(created_at, id)`, the
same pair the ORDER BY uses, which makes the page boundary exact.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from daikonstudio.domain.shared.pagination import PageResult

__all__ = [
    "DEFAULT_PAGE_SIZE",
    "MAX_PAGE_SIZE",
    "PageResult",
    "clamp_limit",
    "encode_ts_cursor",
    "parse_ts_cursor",
]

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200


def clamp_limit(limit: int | None, *, max_size: int = MAX_PAGE_SIZE) -> int:
    if limit is None:
        return DEFAULT_PAGE_SIZE
    return max(1, min(limit, max_size))


def encode_ts_cursor(timestamp: datetime, id_: uuid.UUID) -> str:
    return f"{timestamp.isoformat()}|{id_}"


def parse_ts_cursor(cursor: str | None) -> tuple[datetime, uuid.UUID] | None:
    """Return `None` for anything unparseable -- a malformed cursor restarts the
    listing rather than erroring; it is an opaque token the client never composes."""
    if not cursor:
        return None
    try:
        timestamp, id_ = cursor.split("|", 1)
        return datetime.fromisoformat(timestamp), uuid.UUID(id_)
    except ValueError:
        return None
