from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class TargetKind(StrEnum):
    NUMERIC = "numeric"
    BINARY = "binary"


# Names the pipeline injects into a frame *downstream* of a Dataset, so a
# TargetSpec choosing one of them collides with that column rather than the
# scientist's own data:
#   - "structure"/"uncertainty"/"applicability" -- the prediction-results
#     columns `predict_with_protocol.py`'s `RunPrediction` always writes.
#   - "generation_method" -- the provenance column/tag `export_collection.py`
#     always appends to a Collection export.
#   - "split" -- the partition label `assign_split.py` always adds, via
#     `with_columns`, which silently overwrites a same-named column rather
#     than refusing to.
#   - "row_id" -- the per-engine row index `infrastructure/engines/_scoring.py`
#     returns from every `predict()` call; not persisted today, reserved
#     defensively since nothing stops a future caller from persisting it.
# A readout is derived 1:1 from `TargetSpec.column` (`derive_readouts.py`), so
# a collision here means the pipeline's own column and the model's predicted
# value silently overwrite one another -- caught once, at Dataset creation,
# rather than downstream where the damage is already served to a client.
RESERVED_TARGET_COLUMNS = frozenset(
    {"structure", "uncertainty", "applicability", "generation_method", "row_id", "split"}
)


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
