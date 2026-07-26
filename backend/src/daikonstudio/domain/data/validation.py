from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class InvalidRow:
    row_number: int
    value: str
    reason: str


@dataclass(frozen=True, kw_only=True)
class ConflictRow:
    structure: str
    values: list[int]


@dataclass(frozen=True, kw_only=True)
class ValidationReport:
    total_rows: int
    valid_rows: int
    invalid: list[InvalidRow] = field(default_factory=list)
    conflicting: list[ConflictRow] = field(default_factory=list)
    duplicates_collapsed: int = 0
    salts_flagged: int = 0
    duplicate_spread: float | None = None
