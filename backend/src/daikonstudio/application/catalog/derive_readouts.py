"""Readout derivation: the mechanism that keeps a prediction honest.

An Engine cannot declare its concrete outputs -- they depend on what it was
trained on. A Protocol's Readouts are instead derived here, from the Dataset's
TargetSpec, at training time. That is how a predicted IC50 arrives carrying
the same unit and the same direction as a measured one, which is what lets a
chemist compare the two without unit archaeology.

This lives in the application layer, not `domain.catalog`, because it needs
both `TargetSpec` (`domain.data`) and `Readout` (`domain.catalog`), and the
bounded-context-independence contract forbids `domain.catalog` from importing
`domain.data`. Application code has no such restriction: the layers contract
lets `application` import `domain` freely, including across its bounded
contexts, and coordinating across the catalog and data contexts is exactly what
an application-layer function is for.

It reads each target's own kind, not a training task, so `TaskType` is not
imported here: one Dataset may mix kinds, and the readouts follow the targets.
"""

from __future__ import annotations

from collections.abc import Sequence

from daikonstudio.domain.catalog.readout import Readout, ReadoutType
from daikonstudio.domain.data.target import Direction, TargetKind, TargetSpec, probability_column


def derive_readouts(targets: Sequence[TargetSpec]) -> tuple[Readout, ...]:
    """Each target's readouts, from that target's own kind, in the Dataset's order.

    From the kind and not from a training task, because one Dataset may mix kinds:
    a solubility value beside a reactivity flag is one numeric readout beside a
    probability/class pair.
    """
    return tuple(readout for target in targets for readout in _readouts_for(target))


def _readouts_for(target: TargetSpec) -> tuple[Readout, ...]:
    direction = target.direction.value if target.direction is not None else None
    if target.kind is TargetKind.NUMERIC:
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
            name=probability_column(target.column),
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


def target_columns_of(readouts: Sequence[Readout]) -> tuple[str, ...]:
    """The target columns a Protocol predicts, recovered from its readouts.

    Every target contributes exactly one readout named after its own column --
    NUMERIC for a measured value, CLASS for a binary one -- and only a binary target
    adds a second, PROBABILITY, readout. So the non-probability readouts, in order,
    are the targets, in order. This is what lets prediction and the Scorecard name a
    Protocol's targets without loading its training Dataset, which `RunPrediction`
    deliberately never does.
    """
    return tuple(r.name for r in readouts if r.type is not ReadoutType.PROBABILITY)
