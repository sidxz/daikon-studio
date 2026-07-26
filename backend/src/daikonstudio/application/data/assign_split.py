"""Partitions a dataset into train/validation/test -- the most scientifically
consequential step in the pipeline. A random split of chemical data lets closely
related analogues land on both sides, so the model is scored on near-copies of what
it saw; a scaffold split keeps every Bemis-Murcko scaffold family on one side, which
is what makes the resulting metrics answer "will this work on unseen chemistry?"

Both strategies are deterministic: identical `(frame, spec.seed)` always produces
identical assignments. RANDOM draws that determinism straight from the seeded
`numpy.random.Generator`. SCAFFOLD's grouping and greedy fill are themselves fully
determined by the input frame, but which row bears the tie-break when groups are
equal in size is not left to file order (a CSV sorted by potency or appended
chronologically would otherwise leak a systematic bias into "random" ties) -- it is
resolved by the seeded RNG instead, so re-sorting the input can't change the split.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.shared.errors import ValidationError

_SINGLETON_PREFIX = "\0singleton-"


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
        scaffold if scaffold != "" else f"{_SINGLETON_PREFIX}{index}"
        for index, scaffold in enumerate(scaffolds)
    ]
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(group_keys):
        groups.setdefault(key, []).append(index)

    # Real scaffold families keep first-occurrence tie order (unaffected by the seed).
    # Singleton groups -- typically the acyclic/invalid rows, the ones most likely to
    # dominate a dataset and the ones an uploaded CSV is most likely to have sorted by
    # something scientifically meaningful (potency, upload date) -- are shuffled by
    # the seeded RNG before the stable size sort, so their tie order comes from the
    # seed, not from wherever they happened to sit in the file.
    is_singleton = {key: key.startswith(_SINGLETON_PREFIX) for key in groups}
    real_groups = [group for key, group in groups.items() if not is_singleton[key]]
    singleton_groups = [group for key, group in groups.items() if is_singleton[key]]
    shuffle_order = np.random.default_rng(spec.seed).permutation(len(singleton_groups))
    singleton_groups = [singleton_groups[i] for i in shuffle_order]
    # A stable sort on size alone leaves ties in the order given -- first-occurrence
    # for real families, seed-shuffled for singletons.
    ordered_groups = sorted(real_groups + singleton_groups, key=len, reverse=True)

    train_n, val_n, test_n = _partition_sizes(n, spec.fractions)
    train_cutoff = train_n
    valid_cutoff = train_n + val_n

    # The standard deterministic scaffold-split construction (as used by DeepChem /
    # MoleculeNet): largest groups first, greedily filling train to its cutoff, then
    # validation, then test. A single scaffold family can legitimately be too large
    # for the partition it would "fairly" belong to (a family holding 90% of the rows
    # cannot fit an 80% train target) -- best-effort fill, never an error for that
    # alone, because refusing to split a workable dataset over one dominant family
    # would block real workflows. The invariant that never bends is that a family
    # never straddles a partition.
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
    test_count = n - train_count - valid_count

    # Best-effort fill can still leave a partition the caller explicitly asked for
    # (a nonzero fraction) completely empty -- e.g. a 40-compound congeneric series
    # scaffold-split 80/10/10 can silently return zero training rows. That is not a
    # lopsided-but-usable split, it is an untrainable dataset handed back looking
    # like a normal one. Fail loudly instead of letting it pass silently downstream.
    realized = {"train": train_count, "validation": valid_count, "test": test_count}
    targets = {"train": train_n, "validation": val_n, "test": test_n}
    empty = [
        name
        for name in ("train", "validation", "test")
        if targets[name] > 0 and realized[name] == 0
    ]
    if empty:
        dominant = ordered_groups[0]
        dominant_scaffold = scaffolds[dominant[0]] or "(acyclic/invalid -- no shared scaffold)"
        share = len(dominant) / n
        raise ValidationError(
            f"Scaffold split cannot honor the requested fractions on this dataset: "
            f"{' and '.join(empty)} would be left empty. The largest scaffold family "
            f"('{dominant_scaffold}') spans {share:.0%} of the {n} rows and cannot be "
            f"divided across partitions without a scaffold straddling them. Use "
            f"SplitStrategy.RANDOM instead, or add more chemically diverse compounds "
            f"to this dataset."
        )
    return labels
