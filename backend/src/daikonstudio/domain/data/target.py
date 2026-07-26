from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TargetKind(StrEnum):
    NUMERIC = "numeric"
    BINARY = "binary"


class Direction(StrEnum):
    HIGH = "high"
    LOW = "low"


@dataclass(frozen=True, kw_only=True)
class TargetSpec:
    """What the scientist is predicting, and in what units.

    Readouts on a trained Protocol are derived from this, which is the mechanism by
    which a predicted IC50 arrives in the same unit and direction as a measured one.
    """

    column: str
    kind: TargetKind
    unit: str | None = None
    direction: Direction | None = None
