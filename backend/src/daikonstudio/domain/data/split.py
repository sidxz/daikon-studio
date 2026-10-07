from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class SplitStrategy(StrEnum):
    RANDOM = "random"
    SCAFFOLD = "scaffold"
    #: Homology clustering (MMseqs2) -- the sequence counterpart of SCAFFOLD.
    IDENTITY = "identity"
    #: Mutated-position holdout, for variants of one parent protein, where every
    #: sequence is homologous to every other and IDENTITY yields a single cluster.
    POSITION = "position"
    # ponytail: Butina, UMAP-cluster and temporal splits are still unimplemented.
    # These four cover the two regimes that exist today -- distinct chemistry or
    # distinct families (SCAFFOLD, IDENTITY) and variants of one parent (POSITION)
    # -- and every one of them but RANDOM makes the Scorecard's optimism gap real.


@dataclass(frozen=True, kw_only=True)
class SplitSpec:
    """A named, visible scientific choice -- never a silent 80/10/10."""

    strategy: SplitStrategy
    seed: int
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)

    def __post_init__(self) -> None:
        if abs(sum(self.fractions) - 1.0) > 1e-6:
            raise ValueError("Split fractions must sum to 1.")
        if any(fraction < 0 for fraction in self.fractions):
            raise ValueError("Split fractions must be non-negative.")


def split_to_dict(split: SplitSpec) -> dict[str, Any]:
    return {
        "strategy": split.strategy.value,
        "seed": split.seed,
        "fractions": list(split.fractions),
    }


def split_from_dict(data: Mapping[str, Any]) -> SplitSpec:
    train, validation, test = data["fractions"]
    return SplitSpec(
        strategy=SplitStrategy(data["strategy"]),
        seed=data["seed"],
        fractions=(train, validation, test),
    )
