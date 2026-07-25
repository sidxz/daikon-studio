"""The Engine contract's declarative half.

Constrained exactly as far as prot-cellar's ParamField: enough that the frontend
renders the condition form directly, and no further. The manifest is deliberately
free of infrastructure imports so this envelope serializes to HTTP unchanged when
engines move out of process in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TaskType(StrEnum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"


class ConditionType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    ENUM = "enum"
    BOOL = "bool"


@dataclass(frozen=True, kw_only=True)
class ConditionSpec:
    key: str
    label: str
    type: ConditionType
    required: bool = False
    default: object | None = None
    minimum: float | None = None
    maximum: float | None = None
    options: tuple[str, ...] = ()
    help: str | None = None


@dataclass(frozen=True, kw_only=True)
class EngineManifest:
    id: str
    version: str
    name: str
    description: str
    tasks: tuple[TaskType, ...]
    conditions: tuple[ConditionSpec, ...] = ()
    is_baseline: bool = False


def validate_conditions(
    manifest: EngineManifest, supplied: dict[str, object]
) -> dict[str, object]:
    """Fill defaults, reject unknown keys and out-of-range values."""
    known = {c.key: c for c in manifest.conditions}
    unknown = set(supplied) - set(known)
    if unknown:
        raise ValueError(f"unknown conditions for {manifest.id}: {sorted(unknown)}")

    resolved: dict[str, object] = {}
    for key, spec in known.items():
        if key in supplied:
            value = supplied[key]
        elif spec.default is not None:
            value = spec.default
        elif spec.required:
            raise ValueError(f"{key} is required for {manifest.id}")
        else:
            continue

        if spec.minimum is not None or spec.maximum is not None:
            try:
                numeric_value = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be numeric, got {value!r}") from exc
            if spec.minimum is not None and numeric_value < spec.minimum:
                raise ValueError(f"{key} below minimum {spec.minimum}")
            if spec.maximum is not None and numeric_value > spec.maximum:
                raise ValueError(f"{key} above maximum {spec.maximum}")
        if spec.options and value not in spec.options:
            raise ValueError(f"{key} must be one of {spec.options}")
        resolved[key] = value
    return resolved
