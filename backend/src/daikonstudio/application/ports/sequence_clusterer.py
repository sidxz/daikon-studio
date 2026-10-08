from typing import Protocol


class SequenceClusterer(Protocol):
    """Groups amino-acid sequences by homology, for a leakage-aware split.

    A second port rather than a method on `StructureNormalizer`: that port's methods
    are all one RDKit call on SMILES, and its own docstring argues against splitting
    *those* across ports. This is a different capability on a different input type,
    backed by an external binary rather than a Python library, so it gets its own.
    """

    def cluster(self, sequences: list[str], *, min_identity: float, coverage: float) -> list[int]:
        """Cluster ids, one per input sequence, aligned with the input.

        Ids are arbitrary but dense from zero, and two sequences share an id exactly
        when they landed in the same homology cluster. `min_identity` is the sequence
        identity threshold (0-1) and `coverage` the fraction of the longer sequence the
        alignment must span.

        **Must be deterministic**, including across processes and machines: the split it
        feeds promises that identical `(frame, seed)` always yields identical partitions,
        and a clusterer that reshuffles between runs would silently break that promise.
        """
        ...
