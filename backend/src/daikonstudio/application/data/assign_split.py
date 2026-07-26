"""Partitions a dataset into train/validation/test -- the most scientifically
consequential step in the pipeline. A random split of chemical data lets closely
related analogues land on both sides, so the model is scored on near-copies of what
it saw; a scaffold split keeps every Bemis-Murcko scaffold family on one side, which
is what makes the resulting metrics answer "will this work on unseen chemistry?"

Both strategies are deterministic: identical `(frame, spec.seed)` always produces
identical assignments. RANDOM draws that determinism straight from the seeded
`numpy.random.Generator`; SCAFFOLD needs no randomness at all -- grouping by scaffold
and sorting by descending size is already fully determined by the input frame, so the
seed is simply unused on that path.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy


def assign_split(
    frame: pl.DataFrame,
    structure_column: str,
    spec: SplitSpec,
    normalizer: StructureNormalizer,
) -> pl.DataFrame:
    if spec.strategy is SplitStrategy.RANDOM:
        labels = _random_labels(frame.height, spec)
    else:
        labels = _scaffold_labels(frame, structure_column, spec, normalizer)
    return frame.with_columns(pl.Series("split", labels))


def _partition_sizes(n: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    """Deterministic target row counts per partition; always sums to exactly `n`.

    Train and validation are rounded independently (each capped by whatever capacity
    remains); test takes what's left. Test is the partition the design intent calls
    the pessimistic, honest one to shortchange when fractions don't divide `n` evenly
    -- and with fewer rows than partitions, this is also what pins the tiny-frame
    case: the largest fraction (train) absorbs the rows, and the others land on 0.
    """
    train_n = min(round(n * fractions[0]), n)
    val_n = min(round(n * fractions[1]), n - train_n)
    test_n = n - train_n - val_n
    return train_n, val_n, test_n


def _random_labels(n: int, spec: SplitSpec) -> list[str]:
    if n == 0:
        return []
    train_n, val_n, _test_n = _partition_sizes(n, spec.fractions)
    order = np.random.default_rng(spec.seed).permutation(n)
    labels = [""] * n
    for position, row in enumerate(order):
        if position < train_n:
            labels[row] = "train"
        elif position < train_n + val_n:
            labels[row] = "validation"
        else:
            labels[row] = "test"
    return labels


def _scaffold_labels(
    frame: pl.DataFrame,
    structure_column: str,
    spec: SplitSpec,
    normalizer: StructureNormalizer,
) -> list[str]:
    n = frame.height
    if n == 0:
        return []
    scaffolds = [normalizer.murcko_scaffold(str(s)) for s in frame[structure_column].to_list()]

    # An empty scaffold means "acyclic or invalid", not "same family" -- chemically,
    # two acyclic molecules share nothing just because neither has a ring. Lumping
    # every such row into one mega-group would let that coincidence dictate an entire
    # partition; each empty-scaffold row instead gets its own singleton group, keyed
    # uniquely by row position so it can never collide with another row's key.
    group_keys = [
        scaffold if scaffold != "" else f"\0singleton-{index}"
        for index, scaffold in enumerate(scaffolds)
    ]
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(group_keys):
        groups.setdefault(key, []).append(index)
    # dict preserves first-insertion order, so a stable sort on size alone leaves ties
    # in first-occurrence order -- deterministic without touching the RNG.
    ordered_groups = sorted(groups.values(), key=len, reverse=True)

    train_n, val_n, _test_n = _partition_sizes(n, spec.fractions)
    train_cutoff = train_n
    valid_cutoff = train_n + val_n

    # The standard deterministic scaffold-split construction (as used by DeepChem /
    # MoleculeNet): largest groups first, greedily filling train to its cutoff, then
    # validation, then test. A single scaffold family can legitimately be too large
    # for the partition it would "fairly" belong to (a family holding 90% of the rows
    # cannot fit an 80% train target) -- best-effort fill, never an error, because
    # refusing to split a workable dataset over one dominant family would block real
    # workflows. The invariant that never bends is that a family never straddles.
    labels = [""] * n
    train_count = 0
    valid_count = 0
    for group in ordered_groups:
        size = len(group)
        if train_count + size <= train_cutoff:
            target = "train"
            train_count += size
        elif train_count + valid_count + size <= valid_cutoff:
            target = "validation"
            valid_count += size
        else:
            target = "test"
        for index in group:
            labels[index] = target
    return labels
