"""What a trained Protocol actually predicts.

Deliberately import-free of `domain.data`: `unit` and `direction` cross into a
`Readout` as plain strings rather than as the `Direction` enum the Dataset's
TargetSpec uses, exactly the way `target.py`'s own `target_to_dict` already
flattens `Direction` to `.value` at its JSONB boundary. That keeps this module
pure `domain.catalog`, which the bounded-context-independence contract
requires -- it may not import `domain.data`.

`derive_readouts`, the function that builds these from a Dataset's
TargetSpec, cannot live here for the same reason plus one more: it also needs
`TaskType` from `application.engines`, and no domain module may import
application code at all. See `application/catalog/derive_readouts.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ReadoutType(StrEnum):
    NUMERIC = "numeric"
    PROBABILITY = "probability"
    CLASS = "class"


@dataclass(frozen=True, kw_only=True)
class Readout:
    name: str
    type: ReadoutType
    unit: str | None
    direction: str | None
    description: str
    # The probability at or above which a CLASS readout reads 1. None means 0.5 -- every
    # protocol trained before cutoffs could be tuned, and every untuned one since.
    threshold: float | None = None
