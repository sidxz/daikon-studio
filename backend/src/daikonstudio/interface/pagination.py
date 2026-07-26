"""HTTP-facing pagination wrapper. The cursor helpers themselves live in
`application/pagination.py`, where the queries that use them can reach them."""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["PaginatedResponse"]


class PaginatedResponse[T](BaseModel):
    items: list[T]
    next_cursor: str | None = None
    total_count: int | None = None
