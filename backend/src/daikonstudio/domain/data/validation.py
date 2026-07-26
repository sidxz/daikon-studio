from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from daikonstudio.domain.shared.errors import ValidationError


@dataclass(frozen=True, kw_only=True)
class InvalidRow:
    row_number: int
    value: str
    reason: str


@dataclass(frozen=True, kw_only=True)
class ConflictRow:
    structure: str
    values: list[int]
    # 1-indexed positions in the *uploaded* file, matching `InvalidRow.row_number`'s
    # own convention -- a conflict spans several rows (the replicate measurements
    # that disagree), so this is a collection rather than a single int. Without it,
    # a scientist reading "CCO: [0, 1]" in a ten-thousand-row CSV has no way to find
    # the rows being talked about -- especially since `structure` here is the
    # canonicalized SMILES, which need not match the text the file actually contains.
    row_numbers: list[int]


@dataclass(frozen=True, kw_only=True)
class ValidationReport:
    total_rows: int
    valid_rows: int
    invalid: list[InvalidRow] = field(default_factory=list)
    conflicting: list[ConflictRow] = field(default_factory=list)
    duplicates_collapsed: int = 0
    salts_flagged: int = 0
    duplicate_spread: float | None = None


def report_to_dict(report: ValidationReport) -> dict[str, Any]:
    """Flatten a report for JSONB storage and for the HTTP error body."""
    return asdict(report)


def report_from_dict(data: Mapping[str, Any]) -> ValidationReport:
    return ValidationReport(
        total_rows=data["total_rows"],
        valid_rows=data["valid_rows"],
        invalid=[InvalidRow(**row) for row in data.get("invalid", [])],
        conflicting=[ConflictRow(**row) for row in data.get("conflicting", [])],
        duplicates_collapsed=data.get("duplicates_collapsed", 0),
        salts_flagged=data.get("salts_flagged", 0),
        duplicate_spread=data.get("duplicate_spread"),
    )


class InvalidDatasetError(ValidationError):
    """A rejection that hands back the entire report, not just a message.

    Which rows failed and why, how many duplicates collapsed, how wide the assay
    spread was -- that is the whole value of the validation pass. A bare "invalid
    dataset" would throw it away at the last step and leave the scientist to guess
    which of ten thousand rows to fix.
    """

    def __init__(self, message: str, *, report: ValidationReport) -> None:
        self.report = report
        super().__init__(message)

    def body_extras(self) -> dict[str, Any]:
        return {"detail": report_to_dict(self.report)}
