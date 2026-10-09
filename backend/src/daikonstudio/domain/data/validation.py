from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from daikonstudio.domain.data.structure_kind import StructureKind
from daikonstudio.domain.shared.errors import ValidationError


@dataclass(frozen=True, kw_only=True)
class InvalidRow:
    row_number: int
    value: str
    reason: str


@dataclass(frozen=True, kw_only=True)
class ConflictRow:
    structure: str
    # A compound can conflict in one binary target and agree in another, and the
    # file is fixed in that column, so a conflict has to say which one it is in.
    column: str
    # Binary labels are ints; a disagreement about a predefined split's partition
    # carries the partition names as text. Both are display-only, and
    # `report_from_dict`'s `ConflictRow(**row)` reads either shape back, so widening
    # costs nothing and avoids a second near-identical type.
    values: list[int] | list[str]
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
    # Keyed by target column. Only numeric targets with at least one replicate group
    # appear: a binary target has no spread, and a column with no replicates has
    # nothing to measure, which is different from a spread of zero.
    duplicate_spread: dict[str, float] = field(default_factory=dict)
    # Keyed by target column: how many surviving rows actually carry a measurement for
    # it. Equal to `valid_rows` for every dense dataset, which is every dataset frozen
    # before sparse labels existed -- hence the empty default, which reads back as
    # "not recorded" rather than as zero measured.
    labelled_rows: dict[str, int] = field(default_factory=dict)
    # What the structure column turned out to hold. It belongs here rather than on a
    # column of its own because it is a finding *about the uploaded file* -- in the same
    # family as how many rows were unreadable -- and because it is what decided how every
    # row above was validated. Defaulting to MOLECULE is what makes every dataset frozen
    # before this existed read back correctly.
    structure_kind: StructureKind = StructureKind.MOLECULE
    #: Whether replicate rows of one structure were collapsed into one. False is the
    #: reproduction case: a published benchmark's row count is part of what is being
    #: reproduced, and averaging two measurements into one row makes this a different
    #: dataset from theirs. It belongs here for the same reason `structure_kind` does --
    #: it is a finding about how the uploaded file was read -- and defaulting to True is
    #: what makes every dataset frozen before the toggle read back correctly, because
    #: every one of them was deduplicated.
    deduplicated: bool = True


def report_to_dict(report: ValidationReport) -> dict[str, Any]:
    """Flatten a report for JSONB storage and for the HTTP error body."""
    data = asdict(report)
    # `asdict` leaves the enum member in place. It is a `str` subclass so most encoders
    # cope, but what lands in JSONB should be the plain value rather than whichever
    # encoder happens to serialize it.
    data["structure_kind"] = report.structure_kind.value
    return data


def report_from_dict(data: Mapping[str, Any]) -> ValidationReport:
    return ValidationReport(
        total_rows=data["total_rows"],
        valid_rows=data["valid_rows"],
        invalid=[InvalidRow(**row) for row in data.get("invalid", [])],
        conflicting=[ConflictRow(**row) for row in data.get("conflicting", [])],
        duplicates_collapsed=data.get("duplicates_collapsed", 0),
        salts_flagged=data.get("salts_flagged", 0),
        duplicate_spread=dict(data.get("duplicate_spread") or {}),
        structure_kind=StructureKind(data.get("structure_kind", StructureKind.MOLECULE)),
        deduplicated=data.get("deduplicated", True),
        # `.get` with an empty default, like every field above: a manifest written
        # before sparse labels has no such key, and an absent count reads as "not
        # recorded" rather than as zero measured. Omitting this line entirely is not a
        # harmless oversight -- `ConfirmDatasetPreview` rebuilds the Dataset through
        # here, so the count would be dropped at the moment of freezing and the runner
        # envelope would carry `{}` forever.
        labelled_rows=dict(data.get("labelled_rows") or {}),
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
