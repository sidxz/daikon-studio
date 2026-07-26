from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


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


def target_to_dict(target: TargetSpec) -> dict[str, Any]:
    """Plain JSON, not `asdict`: the enums must land as their string values so a
    stored spec reads the same whether it came from JSONB or from an HTTP body."""
    return {
        "column": target.column,
        "kind": target.kind.value,
        "unit": target.unit,
        "direction": target.direction.value if target.direction else None,
    }


def target_from_dict(data: Mapping[str, Any]) -> TargetSpec:
    direction = data.get("direction")
    return TargetSpec(
        column=data["column"],
        kind=TargetKind(data["kind"]),
        unit=data.get("unit"),
        direction=Direction(direction) if direction else None,
    )
