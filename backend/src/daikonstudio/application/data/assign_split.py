"""Partitions a dataset into train/validation/test -- the most scientifically
consequential step in the pipeline. A random split of chemical data lets closely
related analogues land on both sides, so the model is scored on near-copies of what
it saw; a scaffold split keeps every Bemis-Murcko scaffold family on one side, which
is what makes the resulting metrics answer "will this work on unseen chemistry?"

Three of the four strategies are that same shape -- group the rows, then never let a
group straddle a partition -- and they differ only in what counts as a group:

* SCAFFOLD: the Bemis-Murcko scaffold. Distinct chemistry.
* IDENTITY: an MMseqs2 homology cluster. Distinct protein families.
* POSITION: a connected component of co-mutated residue positions, for variants of
  one parent protein, where every sequence is homologous to every other and IDENTITY
  would hand back a single cluster.

Every strategy is deterministic: identical `(frame, spec.seed)` always produces
identical assignments, including across processes. RANDOM draws that determinism
straight from the seeded `numpy.random.Generator`. The grouped strategies' grouping
(which rows share a group) and greedy fill (which groups go to which partition once
ordered) are fully determined by the input frame's content, never its row order. The one
place row order could matter is when two groups tie in size -- a real scaffold
family of 3 next to another real scaffold family of 3, say, or (the common case,
since most drug-like molecules have a ring and so mostly form real singleton
families rather than the acyclic/invalid kind) any two distinct-scaffold molecules.
That tie is broken by a key derived from the group's own identity and
`spec.seed` via sha256 -- not from which one the file listed first -- so re-sorting
or reversing the uploaded rows cannot change the split for a fixed seed, and
`hash()` (salted per-process by `PYTHONHASHSEED`) is never used for it.

Two places in the grouped strategies would otherwise have smuggled row order back in,
and both are pinned: `SequenceClusterer` promises cluster ids numbered by first
appearance in *input* order (see its port docstring), and POSITION resolves both its
consensus residue and its component identity by taking a minimum rather than a first
-- a modal-residue tie or a union-find root would otherwise depend on which row the
file listed first.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Sequence
from typing import assert_never

import numpy as np
import polars as pl

from daikonstudio.application.data.prepare_frame import RowProgress, map_rows
from daikonstudio.application.ports.sequence_clusterer import SequenceClusterer
from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.shared.errors import ValidationError

_SINGLETON_PREFIX = "\0singleton-"

#: Measured on real ProteinGym assays (see `infrastructure/protein/cluster.py`): at
#: these thresholds 33,069 sequences drawn from 9 unrelated proteins recover exactly
#: those 9 families. Looser and unrelated families merge; tighter and one family
#: fragments, which is the failure that looks fine and leaks.
_MIN_IDENTITY = 0.3
_COVERAGE = 0.8


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
    on_row: RowProgress | None = None,
    clusterer: SequenceClusterer | None = None,
) -> pl.DataFrame:
    # Exhaustive on purpose, and `assert_never` is the whole point of it. This used to
    # be `if RANDOM: ... else: scaffold`, so adding a strategy to the enum silently
    # scaffold-split the data while the Dataset, the Scorecard and the UI all reported
    # the strategy the scientist actually asked for -- a wrong split is not a bug that
    # announces itself. Now an unhandled member fails mypy here, and fails loudly at
    # runtime if it somehow gets past that.
    match spec.strategy:
        case SplitStrategy.RANDOM:
            labels = _random_labels(frame.height, spec)
        case SplitStrategy.SCAFFOLD:
            labels = _scaffold_labels(frame, structure_column, spec, normalizer, on_row)
        case SplitStrategy.IDENTITY:
            labels = _identity_labels(frame, structure_column, spec, clusterer)
        case SplitStrategy.POSITION:
            labels = _position_labels(frame, structure_column, spec)
        case _:  # pragma: no cover - unreachable while the match stays exhaustive
            assert_never(spec.strategy)
    # "split" is one of `domain.data.target.RESERVED_TARGET_COLUMNS` (C1,
    # whole-branch review): `with_columns` below silently overwrites any
    # existing same-named column, including a target's, which is exactly why
    # `CreateDataset` refuses a TargetSpec named "split" before this ever
    # runs. If this ever adds a second injected column, reserve that name too.
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
    on_row: RowProgress | None = None,
) -> list[str]:
    n = frame.height
    if n == 0:
        return []
    structures = [str(s) for s in frame[structure_column].to_list()]
    scaffolds = map_rows(normalizer.murcko_scaffold, structures, on_row)

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

    def describe_dominant(dominant: list[int]) -> str:
        scaffold = scaffolds[dominant[0]] or "(acyclic or unparseable: no ring system)"
        return (
            f"The largest Bemis–Murcko scaffold ('{scaffold}') covers "  # noqa: RUF001
            f"{len(dominant) / n:.0%} of {n} compounds and cannot be divided between "
            f"sets. Use a random split, or add structurally diverse compounds."
        )

    return _grouped_labels(group_keys, spec, describe_dominant)


def _grouped_labels(
    group_keys: Sequence[str],
    spec: SplitSpec,
    describe_dominant: Callable[[list[int]], str],
) -> list[str]:
    """The fill every grouped strategy shares: rows that share a group key land in the
    same partition, whatever made them a group.

    `group_keys` is one key per row, keyed by the row's own *content* so that neither
    the ordering below nor the tie-break can see where a row sat in the file.
    `describe_dominant` turns the largest group's row indices into the operator-facing
    explanation of why the requested fractions were unreachable -- the one part of the
    failure that is strategy-specific, and the main way a scientist discovers they
    picked a strategy that does not suit their data.
    """
    n = len(group_keys)
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(group_keys):
        groups.setdefault(key, []).append(index)

    # Sort by descending size; ties -- any two groups of equal size, real scaffold
    # families and real homology clusters included, not just the singleton kind --
    # are broken by `_tie_break_key`, a function of the group's own identity and the
    # seed. Nothing here depends on which group the file happened to list first:
    # reversing or re-sorting the input rows cannot change the split for a fixed seed.
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
    # validation, then test. A single group can legitimately be too large
    # for the partition it would "fairly" belong to (a group holding 90% of the rows
    # cannot fit an 80% train target) -- best-effort fill, never an error for that
    # alone, because refusing to split a workable dataset over one dominant group
    # would block real workflows. The invariant that never bends is that a group
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
        # "An identity split", not "A identity split" -- this sentence is read by users,
        # and `identity` is the one strategy name that starts with a vowel.
        article = "An" if spec.strategy.value[0] in "aeiou" else "A"
        raise ValidationError(
            f"{article} {spec.strategy.value} split cannot meet the requested fractions "
            f"for this dataset: the {' and '.join(empty)} "
            f"{'set' if len(empty) == 1 else 'sets'} would be empty. "
            + describe_dominant(ordered_groups[0])
        )
    return labels


def _identity_labels(
    frame: pl.DataFrame,
    structure_column: str,
    spec: SplitSpec,
    clusterer: SequenceClusterer | None,
) -> list[str]:
    """Homology clustering: no sequence family straddles a partition.

    The sequence counterpart of `_scaffold_labels`. Leakage between homologous
    sequences is the single largest source of overstated protein-model performance,
    and it persists at identity thresholds as low as 0.2 -- so the group is a
    homology cluster, not an exact-sequence match.
    """
    n = frame.height
    if n == 0:
        return []
    if clusterer is None:
        # Reachable only from a caller that forgot to wire the port through -- the
        # container and both call sites do. Still an operator-facing message rather
        # than an assert, because the operator is who can act on it.
        raise ValidationError(
            "An identity-clustered split needs the sequence-clustering tool, which is "
            "not available in this process. An administrator can check the server's "
            "configuration, or you can choose a different split strategy."
        )
    sequences = [str(s) for s in frame[structure_column].to_list()]
    cluster_ids = clusterer.cluster(sequences, min_identity=_MIN_IDENTITY, coverage=_COVERAGE)

    def describe_dominant(dominant: list[int]) -> str:
        # The common way to land here is the single-parent variant series: every
        # sequence is a near-copy of the same protein, so MMseqs2 correctly returns
        # ONE cluster covering 100% of the rows and no partition is possible. That is
        # the clusterer working, not failing, and the useful thing to say is which
        # strategy does suit that data -- this error is how a scientist finds out.
        return (
            f"The largest group of homologous sequences covers {len(dominant) / n:.0%} "
            f"of {n} sequences and cannot be divided between sets. If these are "
            f"variants of a single parent protein then they are all homologous to one "
            f"another, and no identity threshold can separate them -- use a position "
            f"split, which holds out mutated positions instead. Otherwise add sequences "
            f"from unrelated protein families, or use a random split."
        )

    # Key each cluster by its own alphabetically-first sequence, never by the clusterer's
    # id. The ids are dense from zero in order of first appearance, so they are a function
    # of row position -- and `_grouped_labels` feeds the group key to `_tie_break_key`
    # whenever two groups are the same size. Keying by id therefore moved the split when
    # the uploaded rows were re-sorted, which is precisely what this module promises never
    # happens. It hid behind a size-ordered test: a dataset of unrelated sequences is all
    # singleton clusters, so *every* pair ties and the key decides everything.
    anchors: dict[int, str] = {}
    for cluster, sequence in zip(cluster_ids, sequences, strict=True):
        current = anchors.get(cluster)
        if current is None or sequence < current:
            anchors[cluster] = sequence
    return _grouped_labels([anchors[cluster] for cluster in cluster_ids], spec, describe_dominant)


def _position_labels(
    frame: pl.DataFrame,
    structure_column: str,
    spec: SplitSpec,
) -> list[str]:
    """Mutated-position holdout, for variants of one parent protein.

    IDENTITY cannot help here: every variant is homologous to every other, so it
    yields a single cluster. What *can* leak instead is a position -- train on the
    V5A variant and test on V5L and the model has already seen that site vary.

    The group is therefore a **connected component of co-mutated positions**, not a
    position set. Grouping by the position set is the obvious version and it leaks:
    a single mutant at 5 and a double mutant at (5, 100) would form two different
    groups, so position 5 could be held out in test and trained on in train at the
    same time. Union-find merges any positions that co-occur in a single variant, so
    a position belongs to exactly one group and therefore exactly one partition.
    On the common single-substitution series every component is one position, which
    is the intended simple case; multi-mutants coarsen the grouping instead of
    leaking, which is the trade this makes on purpose.
    """
    n = frame.height
    if n == 0:
        return []
    sequences = [str(s) for s in frame[structure_column].to_list()]
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) > 1:
        # Nothing below means anything on unaligned sequences: "position 37" is not the
        # same site in two sequences of different length, so the components would merge
        # unrelated sites and the holdout would be meaningless rather than wrong-looking.
        raise ValidationError(
            f"A position split needs an aligned single-parent variant series: every "
            f"sequence must have the same length, but these range from {min(lengths)} "
            f"to {max(lengths)} residues. Use an identity split for sequences from "
            f"different proteins, or align the series to a common parent first."
        )

    consensus = _consensus(sequences)
    # ponytail: O(rows x residues) char comparison in Python -- ~10M compares at the
    # 33k x 300 shape measured in `infrastructure/protein/cluster.py`, a few seconds
    # once per dataset build. Vectorize with numpy if a series an order larger shows up.
    mutated = [
        frozenset(
            index
            for index, (residue, parent) in enumerate(zip(sequence, consensus, strict=True))
            if residue != parent
        )
        for sequence in sequences
    ]
    component_of = _co_mutation_components(mutated)
    group_keys = [
        f"position-component-{component_of[min(positions)]}"
        if positions
        # No differences from the consensus: this is the wild type. It belongs to no
        # position, so it is its own group -- the same call `_scaffold_labels` makes for
        # an empty Murcko scaffold, where "no ring system" means singleton, not "same
        # family". Keyed by the sequence text, not the row index, so it cannot
        # reintroduce file-order dependence (and a duplicate wild-type row merges).
        else f"{_SINGLETON_PREFIX}{sequence}"
        for positions, sequence in zip(mutated, sequences, strict=True)
    ]

    def describe_dominant(dominant: list[int]) -> str:
        return (
            f"The largest group of variants mutating the same position covers "
            f"{len(dominant) / n:.0%} of {n} sequences and cannot be divided between "
            f"sets -- a position held out in one set must not be trained on in another. "
            f"Add variants at more distinct positions, or use a random split."
        )

    return _grouped_labels(group_keys, spec, describe_dominant)


def _consensus(sequences: list[str]) -> str:
    """The parent sequence: the modal residue at each position.

    Inferred rather than asked for, because the parent is not a field any upload
    carries. A tie is broken by the residue letter, not by `Counter.most_common`'s
    insertion order -- which row the file listed first must not decide what counts as
    the parent, or the whole split moves when the CSV is re-sorted.
    """
    return "".join(
        max(Counter(column).items(), key=lambda item: (item[1], item[0]))[0]
        for column in zip(*sequences, strict=True)
    )


def _co_mutation_components(mutated: list[frozenset[int]]) -> dict[int, int]:
    """Union-find over positions: `{position: component id}`.

    Positions that co-occur in any one variant end up in the same component. Unions
    always point the larger root at the smaller, so a component's id is its minimum
    position regardless of the order the rows arrived in -- the root of a naive
    union-find depends on union order, which would make the group keys (and so the
    tie-break, and so the split) depend on row order.
    """
    parent: dict[int, int] = {}

    def find(position: int) -> int:
        parent.setdefault(position, position)
        root = position
        while parent[root] != root:
            root = parent[root]
        while parent[position] != root:  # path compression
            parent[position], position = root, parent[position]
        return root

    for positions in mutated:
        if not positions:
            continue
        anchor = min(positions)
        for position in positions:
            left, right = find(anchor), find(position)
            if left != right:
                parent[max(left, right)] = min(left, right)
    return {position: find(position) for position in list(parent)}
