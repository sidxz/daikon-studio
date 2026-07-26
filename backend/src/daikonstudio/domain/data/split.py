from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SplitStrategy(StrEnum):
    RANDOM = "random"
    SCAFFOLD = "scaffold"
    # ponytail: Butina, UMAP-cluster and temporal splits land in Phase 2. Two
    # strategies is the minimum that makes the optimism gap on the Scorecard real.


@dataclass(frozen=True, kw_only=True)
class SplitSpec:
    """A named, visible scientific choice -- never a silent 80/10/10."""

    strategy: SplitStrategy
    seed: int
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)

    def __post_init__(self) -> None:
        if abs(sum(self.fractions) - 1.0) > 1e-6:
            raise ValueError("split fractions must sum to 1.0")
        if any(fraction < 0 for fraction in self.fractions):
            raise ValueError("split fractions must be non-negative")
