from typing import Protocol


class StructureNormalizer(Protocol):
    def canonicalize(self, smiles: str) -> str | None: ...
    def has_multiple_components(self, smiles: str) -> bool: ...
    def murcko_scaffold(self, smiles: str) -> str: ...

    # Not single-structure like the three methods above, but the same
    # "RDKit-shaped capability application needs without importing infrastructure"
    # rationale that put `murcko_scaffold` here (Task 10): a second single-method
    # port for one more RDKit function would be a needless abstraction split. The
    # two set-shaped methods below join for the same reason -- the dataset profile
    # needs physicochemical descriptors and a near-neighbour pair scan, and
    # splitting either into its own port would be that same needless split twice.
    def nearest_neighbour_tanimoto(
        self, query: list[str], reference: list[str]
    ) -> list[float]: ...

    def nearest_neighbours_tanimoto(
        self, query: list[str], reference: list[str], k: int
    ) -> tuple[list[list[int]], list[list[float]]]:
        """The `k` nearest reference indices and their Tanimoto, most similar first."""
        ...

    def descriptors(self, smiles_list: list[str]) -> dict[str, list[float | None]]:
        """Physicochemical descriptors, column-oriented, aligned with the input.

        `None` for a structure that cannot be parsed or a descriptor that cannot
        be computed -- never a substituted zero, which would land in a histogram
        bin and be read as a measurement.
        """
        ...

    def high_similarity_pairs(
        self, structures: list[str], *, threshold: float, limit: int
    ) -> list[tuple[int, int, float]]:
        """Index pairs (`i < j`) at or above `threshold` Tanimoto, strongest first."""
        ...
