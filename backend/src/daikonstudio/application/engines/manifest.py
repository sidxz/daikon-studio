"""The Engine contract's declarative half.

Constrained exactly as far as prot-cellar's ParamField: enough that the frontend
renders the condition form directly, and no further. The manifest is deliberately
free of infrastructure imports so this envelope serializes to HTTP unchanged when
engines move out of process in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# The lane an engine runs in when it does not ask for anything special. A lane is a
# requirement ("this needs a GPU"), not a machine: a deployment satisfies it by running
# a registered runner for that lane, on whatever hardware it has.
DEFAULT_LANE = "default"


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
    # The tasks this setting means anything for; empty means every task. A form hides a
    # setting the dataset has no such task for, and an engine ignores it in a fit of any
    # other task -- one target of a mixed dataset, fanned out.
    tasks: tuple[TaskType, ...] = ()


@dataclass(frozen=True, kw_only=True)
class EngineManifest:
    id: str
    version: str
    name: str
    description: str
    tasks: tuple[TaskType, ...]
    conditions: tuple[ConditionSpec, ...] = ()
    lane: str = DEFAULT_LANE
    is_baseline: bool = False
    # Learns every target of a Dataset in one model. It selects the training path
    # and labels the result ("one joint model" vs "one model per target"); it does
    # not decide which engines a Dataset may use -- every engine accepts every
    # Dataset, the rest through `FanOut`.
    supports_multitask: bool = False


def _coerce(label: str, condition_type: ConditionType, value: object) -> object:
    """Coerce `value` to the Python type `condition_type` declares.

    Values arrive over JSON from a form, where a numeric field can legitimately show
    up as a string (e.g. "500"), so numeric strings are accepted and converted rather
    than rejected. `bool` is excluded from INTEGER/NUMBER explicitly: `isinstance(True,
    int)` is `True` in Python, so without this check a stray boolean would silently
    pass as a number.
    """
    article = "an" if condition_type is ConditionType.INTEGER else "a"
    if condition_type in (ConditionType.INTEGER, ConditionType.NUMBER):
        if isinstance(value, bool):
            raise ValueError(
                f"{label} must be {article} {condition_type.value} (received {value})."
            )
        if isinstance(value, int | float):
            if condition_type is ConditionType.NUMBER:
                return float(value)
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(f"{label} must be an integer (received {value}).")
            return int(value)
        if isinstance(value, str):
            try:
                return int(value) if condition_type is ConditionType.INTEGER else float(value)
            except ValueError as exc:
                raise ValueError(
                    f"{label} must be {article} {condition_type.value} (received {value})."
                ) from exc
        raise ValueError(f"{label} must be {article} {condition_type.value} (received {value}).")
    if condition_type in (ConditionType.STRING, ConditionType.ENUM):
        if isinstance(value, str):
            return value
        raise ValueError(f"{label} must be a string (received {value}).")
    if isinstance(value, bool):
        return value
    raise ValueError(f"{label} must be a boolean (received {value}).")


def validate_conditions(
    manifest: EngineManifest, supplied: dict[str, object]
) -> dict[str, object]:
    """Fill defaults, reject unknown keys, coerce to the declared type, enforce bounds."""
    known = {c.key: c for c in manifest.conditions}
    unknown = set(supplied) - set(known)
    if unknown:
        raise ValueError(
            f"{manifest.name} does not accept these settings: {', '.join(sorted(unknown))}."
        )

    resolved: dict[str, object] = {}
    for key, spec in known.items():
        if key in supplied:
            value = supplied[key]
        elif spec.default is not None:
            value = spec.default
        elif spec.required:
            raise ValueError(f"{spec.label} is required for {manifest.name}.")
        else:
            continue

        value = _coerce(spec.label, spec.type, value)

        if spec.minimum is not None or spec.maximum is not None:
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError(f"{key} must be numeric to enforce bounds, got {value!r}")
            if spec.minimum is not None and value < spec.minimum:
                raise ValueError(f"{spec.label} must be at least {spec.minimum:g}.")
            if spec.maximum is not None and value > spec.maximum:
                raise ValueError(f"{spec.label} must be at most {spec.maximum:g}.")
        if spec.options and value not in spec.options:
            raise ValueError(f"{spec.label} must be one of: {', '.join(spec.options)}.")
        resolved[key] = value
    return resolved


def lane_for(*manifests: EngineManifest) -> str:
    """The lane a Run needs when more than one engine must fit inside it.

    A training Run fits the chosen engine and the baseline in one process, so it
    has to land on a worker that can serve both. Any non-default lane wins over
    the default one, because the default lane is the "no special hardware"
    lane -- a gpu-lane worker can run an ECFP4 fit, but not the reverse.

    ponytail: two lanes, so "the non-default one" is unambiguous. A Run wanting
    two *different* non-default lanes has no home; if a third lane ever exists,
    reject that pair at enqueue rather than silently picking the first.
    """
    return next((m.lane for m in manifests if m.lane != DEFAULT_LANE), DEFAULT_LANE)
