from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from daikonstudio.domain.shared.errors import ValidationError


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
    {
        "structure",
        "uncertainty",
        "applicability",
        "generation_method",
        "row_id",
        "split",
        # Written by RunPrediction beside every scored row (predict_with_protocol.py).
        "input_row",
        "compound_id",
        # The long-format column every engine's `predict()` output carries once a
        # Dataset can hold several targets (`application/engines/fan_out.py`). Never
        # persisted today; reserved for the same defensive reason as `row_id`.
        "target",
    }
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


def probability_column(column: str) -> str:
    """The readout a binary target's P(class=1) is written under. The one definition,
    shared by `derive_readouts` and the collision check below."""
    return f"{column}_probability"


def uncertainty_column(column: str, *, target_count: int) -> str:
    """Where prediction results store one target's per-compound uncertainty.

    Plain `uncertainty` for a one-target Protocol: it is the name every results file
    written before several targets existed already uses, so those stay readable.
    `{column}_uncertainty` beside each target otherwise, because one number per row
    cannot describe four models.
    """
    return "uncertainty" if target_count == 1 else f"{column}_uncertainty"


def prediction_columns(targets: Sequence[TargetSpec]) -> list[str]:
    """Every column a prediction for these targets writes, besides the fixed ones."""
    names: list[str] = []
    for target in targets:
        if target.kind is TargetKind.BINARY:
            names.append(probability_column(target.column))
        names.append(target.column)
        if len(targets) > 1:
            names.append(uncertainty_column(target.column, target_count=len(targets)))
    return names


def check_targets(targets: Sequence[TargetSpec]) -> None:
    """The invariants a Dataset's targets hold, checked once at creation.

    Every one of these is a silent overwrite downstream if it slips through: a
    prediction results frame is a dict of columns, so two derived names that
    collide keep whichever was written last.
    """
    if not targets:
        raise ValidationError("Choose at least one column to predict.")
    columns = [target.column for target in targets]
    for column in columns:
        if column in RESERVED_TARGET_COLUMNS:
            raise ValidationError(
                f"'{column}' cannot be used as a target column",
                detail=(
                    "The application writes a column with this name to prediction "
                    "results and exports. Rename the column in your file. "
                    f"Reserved names: {', '.join(sorted(RESERVED_TARGET_COLUMNS))}."
                ),
            )
    repeated = sorted({column for column in columns if columns.count(column) > 1})
    if repeated:
        raise ValidationError(f"'{repeated[0]}' is chosen more than once as a target.")
    names = prediction_columns(targets)
    clashing = sorted({name for name in names if names.count(name) > 1})
    if clashing:
        raise ValidationError(
            f"Two targets would write the same prediction column, '{clashing[0]}'.",
            detail=(
                "A binary target named x is predicted as x and x_probability, and with "
                "several targets each one also gets x_uncertainty. Rename one of the "
                "columns in your file."
            ),
        )
