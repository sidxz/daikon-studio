"""Domain-layer pagination primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class PageResult[T]:
    """A single page of query results.

    Attributes:
        items: The items in this page.
        next_cursor: Opaque cursor for the next page (``None`` if last page).
        total_count: Optional total result count.
    """

    items: list[T]
    next_cursor: str | None = None
    total_count: int | None = None
