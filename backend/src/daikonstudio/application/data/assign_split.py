"""Partitions a dataset into train/validation/test -- the most scientifically
consequential step in the pipeline. A random split of chemical data lets closely
related analogues land on both sides, so the model is scored on near-copies of what
it saw; a scaffold split keeps every Bemis-Murcko scaffold family on one side, which
is what makes the resulting metrics answer "will this work on unseen chemistry?"

Both strategies are deterministic: identical `(frame, spec.seed)` always produces
identical assignments, including across processes. RANDOM draws that determinism
straight from the seeded `numpy.random.Generator`. SCAFFOLD's grouping (which rows
share a family) and greedy fill (which families go to which partition once ordered)
are fully determined by the input frame's content, never its row order. The one
place row order could matter is when two groups tie in size -- a real scaffold
family of 3 next to another real scaffold family of 3, say, or (the common case,
since most drug-like molecules have a ring and so mostly form real singleton
families rather than the acyclic/invalid kind) any two distinct-scaffold molecules.
That tie is broken by a key derived from the group's own scaffold identity and
`spec.seed` via sha256 -- not from which one the file listed first -- so re-sorting
or reversing the uploaded rows cannot change the split for a fixed seed, and
`hash()` (salted per-process by `PYTHONHASHSEED`) is never used for it.
"""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl

from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.shared.errors import ValidationError

_SINGLETON_PREFIX = "\0singleton-"


def _tie_break_key(seed: int, group_key: str) -> int:
    """A per-group pseudo-random key: stable across processes (sha256, not the
    per-process-salted `hash()`) and a pure function of the group's own identity, not
    of where its rows sat in the file -- so it can't reintroduce the file-order bias
    this exists to eliminate.
    """
    digest = hashlib.sha256(f"{seed}:{group_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


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
    structures = [str(s) for s in frame[structure_column].to_list()]
    scaffolds = [normalizer.murcko_scaffold(structure) for structure in structures]

    # An empty scaffold means "acyclic or invalid", not "same family" -- chemically,
    # two acyclic molecules share nothing just because neither has a ring. Lumping
    # every such row into one mega-group would let that coincidence dictate an entire
    # partition, so each such row is keyed by its own structure text rather than a
    # shared "" key -- two different acyclic molecules land in different groups (an
    # actual duplicate structure still correctly merges into one). Keying by the
    # structure's own content, not by row position, matters below: it's what makes
    # the tie-break key depend only on the molecule, never on where it sat in the file.
    group_keys = [
        scaffold if scaffold != "" else f"{_SINGLETON_PREFIX}{structure}"
        for scaffold, structure in zip(scaffolds, structures, strict=True)
    ]
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(group_keys):
        groups.setdefault(key, []).append(index)

    # Sort by descending size; ties -- any two groups of equal size, real scaffold
    # families included, not just the acyclic/invalid ones -- are broken by
    # `_tie_break_key`, a function of the group's own identity and the seed. Nothing
    # here depends on which group the file happened to list first: reversing or
    # re-sorting the input rows cannot change the split for a fixed seed.
    ordered_groups = [
        rows
        for _key, rows in sorted(
            groups.items(),
            key=lambda item: (-len(item[1]), _tie_break_key(spec.seed, item[0])),
        )
    ]

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
