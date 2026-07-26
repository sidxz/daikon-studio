"""Keyset-cursor helpers shared by paginated queries.

Keyset, not OFFSET: a listing sorted newest-first is exactly where rows get
inserted, and OFFSET pagination silently repeats or skips rows when that happens
mid-scroll. The cursor is opaque to clients but is really `(created_at, id)`, the
same pair the ORDER BY uses, which makes the page boundary exact.

"Opaque" has to mean transport-safe, not just unreadable, so the payload is
base64url-encoded. The plain `datetime.isoformat()` of an aware timestamp
contains a `+`, which a query-string parser decodes as a space -- the cursor then
fails to parse on the way back in. Combined with a parse failure that quietly
returned `None`, which the repository reads as "start from the beginning", that
turned a client looping on `next_cursor` into an infinite loop serving page one
forever. Hence both halves of the fix: an alphabet with nothing to mangle
(`A-Z a-z 0-9 - _ =`), and a malformed cursor that raises instead of restarting.
"""

from __future__ import annotations

import base64
import uuid
from datetime import datetime

from daikonstudio.domain.shared.errors import ValidationError
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
    return base64.urlsafe_b64encode(f"{timestamp.isoformat()}|{id_}".encode()).decode()


def parse_ts_cursor(cursor: str | None) -> tuple[datetime, uuid.UUID] | None:
    """`None` only for no cursor at all. Anything else that fails to parse raises.

    Silently restarting the listing on a corrupt cursor is how the infinite loop
    hid: the client asks for page two, is handed page one, and never reaches the
    end. A client that mangles the token needs to be told so.

    Every failure mode below is a `ValueError` subclass -- `binascii.Error` from
    bad base64 padding, `UnicodeDecodeError` from non-UTF-8 payload, the unpack
    when the `|` is missing, and both parsers.
    """
    if not cursor:
        return None
    try:
        timestamp, id_ = base64.urlsafe_b64decode(cursor).decode().split("|", 1)
        return datetime.fromisoformat(timestamp), uuid.UUID(id_)
    except ValueError as error:
        raise ValidationError(
            "Invalid pagination cursor",
            detail="Pass back the `next_cursor` from the previous page unmodified.",
        ) from error
