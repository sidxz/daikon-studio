"""Readout derivation: the mechanism that keeps a prediction honest.

An Engine cannot declare its concrete outputs -- they depend on what it was
trained on. A Protocol's Readouts are instead derived here, from the Dataset's
TargetSpec, at training time. That is how a predicted IC50 arrives carrying
the same unit and the same direction as a measured one, which is what lets a
chemist compare the two without unit archaeology.

This lives in the application layer, not `domain.catalog`, because it needs
both `TargetSpec` (`domain.data`) and `TaskType` (`application.engines`) as
parameter types, and neither constraint has a domain-layer answer:

- The bounded-context-independence contract forbids `domain.catalog` from
  importing `domain.data` -- so even taking `TargetSpec` alone rules out a
  home in `domain.catalog`.
- The Clean Architecture layers contract forbids *any* domain module from
  importing `application` code -- so `TaskType` alone rules out a home
  anywhere in `domain`, independent of the bounded-context question.

Application code has neither restriction: the layers contract lets
`application` import `domain` freely (including across its bounded
contexts), and the independence contract only names `domain.*` packages.
Coordinating across the catalog and data contexts is exactly what an
application-layer function is for.
"""

from __future__ import annotations

from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.domain.catalog.readout import Readout, ReadoutType
from daikonstudio.domain.data.target import Direction, TargetSpec


def derive_readouts(target: TargetSpec, task: TaskType) -> tuple[Readout, ...]:
    direction = target.direction.value if target.direction is not None else None

    if task is TaskType.REGRESSION:
        return (
            Readout(
                name=target.column,
                type=ReadoutType.NUMERIC,
                unit=target.unit,
                direction=direction,
                description=f"Predicted {target.column}",
            ),
        )

    return (
        Readout(
            name=f"{target.column}_probability",
            type=ReadoutType.PROBABILITY,
            unit=None,
            direction=Direction.HIGH.value,
            description=f"Probability that {target.column} is positive",
        ),
        Readout(
            name=target.column,
            type=ReadoutType.CLASS,
            unit=None,
            direction=direction,
            description=f"Predicted {target.column} class",
        ),
    )
