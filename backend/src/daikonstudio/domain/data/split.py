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
    #: The partitions are given, not computed: read out of a column in the uploaded
    #: file. This is what makes a published benchmark's own train/test assignment
    #: usable, and so the only way our number and theirs measure the same thing.
    PREDEFINED = "predefined"
    # ponytail: Butina, UMAP-cluster and temporal splits are still unimplemented.
    # These four cover the two regimes that exist today -- distinct chemistry or
    # distinct families (SCAFFOLD, IDENTITY) and variants of one parent (POSITION)
    # -- and every one of them but RANDOM makes the Scorecard's optimism gap real.


#: Named so `__post_init__` can tell "the caller left fractions alone" from "the caller
#: chose something", which is what a predefined split has to refuse.
_DEFAULT_FRACTIONS = (0.8, 0.1, 0.1)

#: The partition names this reads out of an uploaded column, each mapped to the label
#: the rest of the system uses. Lowercased and stripped before lookup, because a column
#: typed by hand or exported from Excel carries "Train " and "TEST".
#:
#: Fold indices (0/1/2) are deliberately absent. A numbered column is a k-fold
#: assignment, which is a different thing from a train/validation/test split: one file
#: describes five experiments, not one. Those are preprocessed into one column per fold
#: outside the app, and the error message for an unrecognised value says so.
_PARTITION_SPELLINGS = {
    "train": "train",
    "training": "train",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "dev": "validation",
    "test": "test",
    "testing": "test",
}


def normalize_partition(value: str) -> str | None:
    """One cell of a predefined-split column as a partition label, or None if it is not
    a partition name at all.

    Domain vocabulary rather than a helper beside its first caller: both the ingestion
    gate (which rejects a bad cell while it still knows the uploaded row number) and
    the splitter (which maps the surviving cells) have to agree on what counts as a
    partition, and a second copy of this table is how they would stop agreeing.
    """
    return _PARTITION_SPELLINGS.get(value.strip().lower())


@dataclass(frozen=True, kw_only=True)
class SplitSpec:
    """A named, visible scientific choice -- never a silent 80/10/10."""

    strategy: SplitStrategy
    seed: int
    fractions: tuple[float, float, float] = _DEFAULT_FRACTIONS
    #: Only for PREDEFINED: the uploaded column holding each row's partition.
    column: str | None = None

    def __post_init__(self) -> None:
        if abs(sum(self.fractions) - 1.0) > 1e-6:
            raise ValueError("Split fractions must sum to 1.")
        if any(fraction < 0 for fraction in self.fractions):
            raise ValueError("Split fractions must be non-negative.")
        if self.strategy is SplitStrategy.PREDEFINED:
            if self.column is None:
                raise ValueError(
                    "A predefined split needs the name of the column holding each row's partition."
                )
            # Fractions are meaningless when the partitions are given, and an inert
            # non-default value would sit on the Dataset looking like it did something.
            if self.fractions != _DEFAULT_FRACTIONS:
                raise ValueError(
                    "Split fractions do not apply to a predefined split: the "
                    "partitions come from the file."
                )
        elif self.column is not None:
            raise ValueError(
                f"A {self.strategy.value} split computes its own partitions and takes no column."
            )


def split_to_dict(split: SplitSpec) -> dict[str, Any]:
    return {
        "strategy": split.strategy.value,
        "seed": split.seed,
        "fractions": list(split.fractions),
        "column": split.column,
    }


def split_from_dict(data: Mapping[str, Any]) -> SplitSpec:
    train, validation, test = data["fractions"]
    return SplitSpec(
        strategy=SplitStrategy(data["strategy"]),
        seed=data["seed"],
        fractions=(train, validation, test),
        # `.get`, not `[...]`: every split frozen before PREDEFINED existed has no such
        # key, and a KeyError here would break loading every one of those datasets.
        column=data.get("column"),
    )
