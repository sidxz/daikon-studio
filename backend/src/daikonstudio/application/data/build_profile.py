"""Derives a `DatasetProfile` from a frozen snapshot.

Pure computation: a frame in, a profile out, chemistry reached through the
`StructureNormalizer` port exactly as `build_scorecard` reaches it. Nothing here
reads or writes storage -- `GetDatasetProfile` next door owns the cache, so this
function stays testable against a bare frame.

Everything is binned before it leaves: a histogram of 24 counts is the same size
whether it came from four hundred compounds or four hundred thousand, and the
consumer draws the bins it is given rather than choosing its own. The one
exception is the top scaffolds and the activity cliffs, which are structures a
chemist looks at individually and would be meaningless as aggregates.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.data.profile import (
    ActivityCliff,
    ClassBalance,
    DatasetProfile,
    DescriptorProfile,
    Histogram,
    NumericSummary,
    ScaffoldEntry,
    ScaffoldProfile,
    SimilarityProfile,
    SplitHistogram,
    Substitution,
    TargetDistribution,
    VariantPosition,
    VariantProfile,
)
from daikonstudio.domain.data.structure_kind import StructureKind
from daikonstudio.domain.data.target import TargetKind, TargetSpec

_HISTOGRAM_BINS = 24

# Fixed 0-1 edges for the similarity histogram, deliberately not data-derived:
# "how close is the test set to the training set" is only interpretable against
# an absolute Tanimoto scale, and bins chosen per dataset would make two datasets
# look alike whenever their *shapes* matched, however far apart their similarities
# actually were.
_SIMILARITY_BINS = 20

# Deliberately the same 0.3 as `application/execution/build_scorecard.py`'s
# `_APPLICABILITY_THRESHOLD`, duplicated rather than imported: `application.data`
# and `application.execution` are separate bounded contexts and do not import each
# other. If one moves, move the other -- the Dataset page's "within domain" and
# the Scorecard's "applicability" are the same measurement asked before and after
# training, and they must not quietly answer different questions.
_WITHIN_DOMAIN_THRESHOLD = 0.3

# A test compound this close to something in the training set is, for a
# fingerprint model, already known. Deliberately short of 1.0: exact duplicates
# cannot survive `prepare_frame`'s dedup, so anything this catches is a *distinct*
# structure that the model will nonetheless recognise.
#
# 0.9 and not the 0.95 this was first written with, because ECFP4 Tanimoto is far
# more discriminating than the intuition behind those numbers. Measured on this
# codebase's own featurizer: a drug-like scaffold and its single-substituent
# analogues (methyl -> ethyl, H -> Cl, sulfonamide -> methylsulfone) score 0.61 to
# 0.82, and paracetamol against its O-methyl ether scores 0.59. A 0.95 threshold
# is effectively unreachable for distinct structures, so it would have reported
# "no near-duplicates" for every dataset ever uploaded -- a constant answer
# presenting as a measurement, which is worse than not reporting it at all.
_NEAR_DUPLICATE_THRESHOLD = 0.9

# Same calibration, one step looser: at 0.75 an ECFP4 pair is a close analogue --
# the regime activity-cliff analysis is about -- while 0.9 would restrict cliffs
# to the handful of near-identical pairs and miss the substituent changes that
# make cliffs interesting. Both thresholds are empirical: re-measure them if the
# fingerprint in `infrastructure/chem/featurize.py` ever changes, because they are
# properties of that featurizer and not of chemistry.
_CLIFF_SIMILARITY = 0.75
_CLIFF_PAIR_LIMIT = 5000
_CLIFF_RESULTS = 8

# The pair scan is O(n^2). Beyond this many compounds it runs on a deterministic
# stride sample instead, and `DatasetProfile.cliffs_sampled_from` records that it
# did -- a truncated search must never present as an exhaustive one.
_MAX_CLIFF_COMPOUNDS = 3000

#: Positions a mutational map will draw. Twenty residues each, so this is the cell
#: budget as much as the position budget; a 1022-residue protein varying everywhere
#: would otherwise put 20,440 cells through the blob store and into a browser.
_MAX_MAP_POSITIONS = 400

_TOP_SCAFFOLDS = 8


def build_profile(
    *,
    frame: pl.DataFrame,
    structure_column: str,
    target: TargetSpec,
    normalizer: StructureNormalizer,
    structure_kind: StructureKind = StructureKind.MOLECULE,
) -> DatasetProfile:
    structures = [str(value) for value in frame[structure_column].to_list()]
    splits = [str(value) for value in frame["split"].to_list()]
    partition_counts = {
        name: sum(1 for split in splits if split == name)
        for name in ("train", "validation", "test")
    }

    train_index = [i for i, split in enumerate(splits) if split == "train"]
    test_index = [i for i, split in enumerate(splits) if split == "test"]

    targets = frame[target.column].to_numpy()
    is_numeric = target.kind is TargetKind.NUMERIC

    # Every chemistry section below reads the structure column as SMILES -- Tanimoto
    # similarity, Bemis-Murcko scaffolds, RDKit descriptors, and the cliff scan that is
    # built on the similarity. On a sequence column each would still return a
    # well-formed answer, and every one of those answers would be meaningless: a
    # fingerprint of unparseable text, a scaffold that does not exist. The profile omits
    # them instead, which is the difference between "not applicable" and a fabricated
    # number that reads as a measurement.
    chemistry = structure_kind is StructureKind.MOLECULE
    descriptors = (
        _descriptors(structures, targets, train_index, test_index, normalizer) if chemistry else []
    )

    return DatasetProfile(
        compounds=frame.height,
        partition_counts=partition_counts,
        target_kind=target.kind.value,
        target_distribution=(
            _target_distribution(targets, train_index, test_index) if is_numeric else None
        ),
        class_balance=[] if is_numeric else _class_balance(targets, splits),
        similarity=(
            _similarity(structures, train_index, test_index, normalizer) if chemistry else None
        ),
        scaffolds=_scaffolds(structures, splits, normalizer) if chemistry else None,
        variants=None if chemistry else _variants(structures, splits, targets),
        descriptors=descriptors,
        best_descriptor=_best_descriptor(descriptors) if chemistry else None,
        activity_cliffs=(
            _activity_cliffs(structures, targets, target, normalizer) if chemistry else []
        ),
        # Reported whenever the scan was subsampled, including when it found
        # nothing: "no cliffs among 3000 of your 12000 compounds" and "no cliffs"
        # are different claims, and only one of them is true here.
        cliffs_sampled_from=(
            frame.height if chemistry and frame.height > _MAX_CLIFF_COMPOUNDS else None
        ),
    )


def _finite(values: np.ndarray) -> np.ndarray:
    """Nulls arrive as NaN through `to_numpy`; a NaN in a histogram silently
    lands nowhere and in a mean poisons every other value, so both are dropped
    here rather than at each call site."""
    numeric = values.astype(np.float64, copy=False)
    return numeric[np.isfinite(numeric)]


def _summary(values: np.ndarray) -> NumericSummary:
    return NumericSummary(
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values)),
    )


def _split_histogram(train: np.ndarray, test: np.ndarray) -> SplitHistogram:
    combined = np.concatenate([train, test]) if train.size or test.size else np.array([0.0])
    edges = np.histogram_bin_edges(combined, bins=_HISTOGRAM_BINS)
    train_counts, _ = np.histogram(train, bins=edges)
    test_counts, _ = np.histogram(test, bins=edges)
    return SplitHistogram(
        edges=[float(edge) for edge in edges],
        train=[int(count) for count in train_counts],
        test=[int(count) for count in test_counts],
    )


def _target_distribution(
    targets: np.ndarray, train_index: list[int], test_index: list[int]
) -> TargetDistribution | None:
    train = _finite(targets[train_index]) if train_index else np.array([])
    test = _finite(targets[test_index]) if test_index else np.array([])
    if train.size == 0 or test.size == 0:
        # One empty partition means there is no train-versus-test comparison to
        # draw, and a one-sided histogram labelled as both would be a lie about
        # which partition it came from.
        return None
    return TargetDistribution(
        histogram=_split_histogram(train, test),
        train=_summary(train),
        test=_summary(test),
    )


def _class_balance(targets: np.ndarray, splits: list[str]) -> list[ClassBalance]:
    balance = []
    for name in ("train", "validation", "test"):
        rows = [targets[i] for i, split in enumerate(splits) if split == name]
        if not rows:
            continue
        # Both classes counted directly over measured rows only. `to_numpy()` yields
        # NaN for an unmeasured target, `nan > 0.5` is False, and `len(rows) -
        # positive` would therefore report every blank as an inactive result -- a
        # falsely imbalanced column presented as a measurement.
        measured = [float(value) for value in rows if not math.isnan(float(value))]
        positive = sum(1 for value in measured if value > 0.5)
        balance.append(
            ClassBalance(split=name, positive=positive, negative=len(measured) - positive)
        )
    return balance


def _similarity(
    structures: list[str],
    train_index: list[int],
    test_index: list[int],
    normalizer: StructureNormalizer,
) -> SimilarityProfile | None:
    if not train_index or not test_index:
        return None
    values = np.array(
        normalizer.nearest_neighbour_tanimoto(
            [structures[i] for i in test_index], [structures[i] for i in train_index]
        )
    )
    if values.size == 0:
        return None
    counts, edges = np.histogram(values, bins=_SIMILARITY_BINS, range=(0.0, 1.0))
    return SimilarityProfile(
        histogram=Histogram(
            edges=[float(edge) for edge in edges], counts=[int(count) for count in counts]
        ),
        median=float(np.median(values)),
        within_domain=float(np.mean(values >= _WITHIN_DOMAIN_THRESHOLD)),
        within_domain_threshold=_WITHIN_DOMAIN_THRESHOLD,
        near_duplicates=int(np.sum(values >= _NEAR_DUPLICATE_THRESHOLD)),
        near_duplicate_threshold=_NEAR_DUPLICATE_THRESHOLD,
    )


def _scaffolds(
    structures: list[str], splits: list[str], normalizer: StructureNormalizer
) -> ScaffoldProfile:
    scaffolds = [normalizer.murcko_scaffold(smiles) for smiles in structures]

    counts: dict[str, int] = {}
    for scaffold in scaffolds:
        counts[scaffold] = counts.get(scaffold, 0) + 1

    # The cross-split check keys acyclic molecules individually, exactly as
    # `assign_split._scaffold_labels` does, and NOT by the shared "" that the
    # display counts above group them under.
    #
    # The two groupings answer different questions and must not be shared. For
    # display, "95 compounds have no ring system" is one useful fact about the
    # dataset. For the integrity check it would be a false alarm: `assign_split`
    # deliberately gives every acyclic molecule its own group -- two acyclic
    # molecules share nothing just because neither has a ring -- so it will
    # happily place them either side of a scaffold split, correctly. Grouping
    # them here reported BBBP's perfectly-behaved scaffold split as leaking 95
    # compounds across one shared scaffold, which is the exact opposite of what
    # this number exists to tell a scientist.
    partitions: dict[str, set[str]] = {}
    split_key_counts: dict[str, int] = {}
    for scaffold, structure, split in zip(scaffolds, structures, splits, strict=True):
        key = scaffold or f"\0acyclic:{structure}"
        partitions.setdefault(key, set()).add(split)
        split_key_counts[key] = split_key_counts.get(key, 0) + 1

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    total = len(scaffolds) or 1

    running = 0
    coverage = []
    for _scaffold, count in ordered:
        running += count
        coverage.append(running / total)

    shared = [
        scaffold
        for scaffold, present in partitions.items()
        if "train" in present and "test" in present
    ]

    return ScaffoldProfile(
        unique_count=len(counts),
        singleton_count=sum(1 for count in counts.values() if count == 1),
        largest_fraction=(ordered[0][1] / total) if ordered else 0.0,
        cumulative_coverage=coverage,
        top=[ScaffoldEntry(smiles=s, count=c) for s, c in ordered[:_TOP_SCAFFOLDS]],
        cross_split_scaffolds=len(shared),
        cross_split_compounds=sum(split_key_counts[scaffold] for scaffold in shared),
    )


def _spearman(values: np.ndarray, targets: np.ndarray) -> float | None:
    """Rank correlation, over the rows where both sides are measurable.

    Spearman rather than Pearson because a descriptor-target relationship in
    chemistry is routinely monotonic and not linear (a logP-solubility trend
    curves), and Pearson would under-report exactly the dependency this number
    exists to expose. It is also the honest choice for a binary target, where it
    reduces to a rank-biserial correlation rather than pretending 0/1 is a scale.
    """
    usable = np.isfinite(values) & np.isfinite(targets)
    if int(np.sum(usable)) < 3:
        return None
    left, right = values[usable], targets[usable]
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        # A constant on either side has no rank order to correlate. `None`, not
        # 0.0, which would read as "measured, and unrelated".
        return None
    correlation = pl.DataFrame({"x": left, "y": right}).select(
        pl.corr("x", "y", method="spearman")
    )[0, 0]
    return None if correlation is None or not np.isfinite(correlation) else float(correlation)


def _descriptors(
    structures: list[str],
    targets: np.ndarray,
    train_index: list[int],
    test_index: list[int],
    normalizer: StructureNormalizer,
) -> list[DescriptorProfile]:
    columns = normalizer.descriptors(structures)
    numeric_targets = targets.astype(np.float64, copy=False)

    profiles = []
    for name, raw in columns.items():
        values = np.array([np.nan if value is None else value for value in raw], dtype=np.float64)
        train = _finite(values[train_index]) if train_index else np.array([])
        test = _finite(values[test_index]) if test_index else np.array([])
        measured = _finite(values)
        if measured.size == 0:
            continue
        profiles.append(
            DescriptorProfile(
                name=name,
                histogram=_split_histogram(train, test),
                median=float(np.median(measured)),
                target_correlation=_spearman(values, numeric_targets),
            )
        )
    return profiles


def _best_descriptor(profiles: list[DescriptorProfile]) -> str | None:
    ranked = [p for p in profiles if p.target_correlation is not None]
    if not ranked:
        return None
    return max(ranked, key=lambda p: abs(p.target_correlation or 0.0)).name


def _activity_cliffs(
    structures: list[str],
    targets: np.ndarray,
    target: TargetSpec,
    normalizer: StructureNormalizer,
) -> list[ActivityCliff]:
    if len(structures) < 2:
        return []

    # A stride sample rather than the first N rows: a CSV is routinely ordered by
    # series, and a head slice would scan one corner of the chemistry and report
    # it as the dataset's cliffs. Deterministic, so a cached profile and a
    # recomputed one agree.
    stride = max(1, -(-len(structures) // _MAX_CLIFF_COMPOUNDS))
    index = list(range(0, len(structures), stride))
    sampled = [structures[i] for i in index]

    pairs = normalizer.high_similarity_pairs(
        sampled, threshold=_CLIFF_SIMILARITY, limit=_CLIFF_PAIR_LIMIT
    )
    if not pairs:
        return []

    numeric_targets = targets.astype(np.float64, copy=False)
    cliffs = []
    for left, right, similarity in pairs:
        left_value = numeric_targets[index[left]]
        right_value = numeric_targets[index[right]]
        if not (np.isfinite(left_value) and np.isfinite(right_value)):
            continue
        delta = abs(float(left_value) - float(right_value))
        if delta == 0.0:
            # Near-identical structures that agree are not a cliff; they are the
            # ordinary case, and the whole set of them would bury the handful
            # that disagree.
            continue
        cliffs.append(
            ActivityCliff(
                left_structure=sampled[left],
                right_structure=sampled[right],
                left_value=float(left_value),
                right_value=float(right_value),
                similarity=similarity,
                delta=delta,
            )
        )

    # A binary target has only one possible non-zero delta, so ranking by it
    # would order the cliffs arbitrarily; the most structurally similar pair is
    # the most striking disagreement instead.
    if target.kind is TargetKind.BINARY:
        cliffs.sort(key=lambda cliff: cliff.similarity, reverse=True)
    else:
        cliffs.sort(key=lambda cliff: cliff.delta, reverse=True)
    return cliffs[:_CLIFF_RESULTS]


def _variants(
    sequences: list[str], splits: list[str], targets: np.ndarray
) -> VariantProfile | None:
    """Which residue positions vary, and how their variants fall across partitions.

    `None` for a ragged series. Equal length is what makes "position 31" mean the same
    thing in every row, and without it a position map would be lining up residues that
    are not comparable -- a picture that reads as a measurement and is not one.
    """
    if not sequences:
        return None
    length = len(sequences[0])
    if length == 0 or any(len(sequence) != length for sequence in sequences):
        return None

    # The parent is the per-position modal residue, with the tie broken on the residue
    # letter so the consensus is a pure function of the data and not of row order --
    # the same rule `assign_split._consensus` uses, and for the same reason.
    columns = list(zip(*sequences, strict=True))
    consensus = "".join(
        max(set(column), key=lambda residue: (column.count(residue), residue))
        for column in columns
    )

    counts: dict[int, dict[str, int]] = {}
    # Keyed by the cell a mutational map draws: one position, one replacement residue.
    # Several rows measuring the same substitution average, which is what the map shows.
    cells: dict[tuple[int, str], list[float]] = {}
    splits_by_cell: dict[tuple[int, str], str] = {}
    unchanged = 0
    multi = 0
    for sequence, split, value in zip(sequences, splits, targets, strict=True):
        differing = [i for i, (a, b) in enumerate(zip(sequence, consensus, strict=True)) if a != b]
        if not differing:
            unchanged += 1
        elif len(differing) > 1:
            multi += 1
        for index in differing:
            tally = counts.setdefault(index, {"train": 0, "validation": 0, "test": 0})
            if split in tally:
                tally[split] += 1
            if np.isfinite(value):
                cells.setdefault((index, sequence[index]), []).append(float(value))
                splits_by_cell[(index, sequence[index])] = split

    # Most-varied positions first when there are more than the map can show, so a
    # truncated map keeps the part of the experiment that was actually explored. The
    # count is reported either way; a partial map that presents as whole is the failure.
    ordered = sorted(counts.items(), key=lambda item: (-sum(item[1].values()), item[0]))
    total_positions = len(ordered)
    truncated = total_positions > _MAX_MAP_POSITIONS
    kept = {index for index, _ in ordered[:_MAX_MAP_POSITIONS]}

    positions = [
        VariantPosition(
            position=index + 1,
            train=tally["train"],
            validation=tally["validation"],
            test=tally["test"],
        )
        for index, tally in sorted(counts.items())
        if index in kept
    ]
    substitutions = [
        Substitution(
            position=index + 1,
            wild_type=consensus[index],
            variant=residue,
            value=sum(values) / len(values),
            split=splits_by_cell[(index, residue)],
        )
        for (index, residue), values in sorted(cells.items())
        if index in kept
    ]
    return VariantProfile(
        consensus=consensus,
        positions=positions,
        unchanged_rows=unchanged,
        multi_mutant_rows=multi,
        held_out_positions=sum(1 for p in positions if p.train == 0),
        substitutions=substitutions,
        positions_sampled_from=total_positions if truncated else None,
    )
