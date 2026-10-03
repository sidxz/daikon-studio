from typing import Protocol


class TooFewCompounds(Exception):
    """Fewer compounds than a 2D layout can say anything about."""


class ChemicalSpaceLayout(Protocol):
    def layout(self, structures: list[str], seed: int) -> tuple[list[float], list[float]]:
        """2D coordinates in [0, 1] for each structure, deterministic for a seed."""
        ...

    def describe(self) -> dict[str, object]:
        """`method`, `params` and library version, recorded with the map."""
        ...
