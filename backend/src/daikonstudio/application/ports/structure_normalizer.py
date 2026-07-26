from typing import Protocol


class StructureNormalizer(Protocol):
    def canonicalize(self, smiles: str) -> str | None: ...
    def has_multiple_components(self, smiles: str) -> bool: ...
    def murcko_scaffold(self, smiles: str) -> str: ...

    # Not single-structure like the three methods above, but the same
    # "RDKit-shaped capability application needs without importing infrastructure"
    # rationale that put `murcko_scaffold` here (Task 10): a second single-method
    # port for one more RDKit function would be a needless abstraction split.
    def nearest_neighbour_tanimoto(
        self, query: list[str], reference: list[str]
    ) -> list[float]: ...
