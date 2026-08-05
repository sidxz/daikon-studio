"""What the Dataset profile must never get wrong.

Not a test per field -- the shapes are mechanical and a broken one is loud. These
cover the four places the profile could quietly tell a scientist something false:
the train/test histograms sharing edges, an unmeasurable number coming back as
`None` rather than a fabricated zero, the split-integrity counts actually
distinguishing a scaffold split from a random one, and a subsampled cliff scan
admitting that it was subsampled.
"""

from __future__ import annotations

import polars as pl
import pytest

from daikonstudio.application.data.build_profile import build_profile
from daikonstudio.domain.data.profile import profile_from_dict, profile_to_dict
from daikonstudio.domain.data.target import TargetKind, TargetSpec
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer

# A congeneric series (one shared diaryl-pyrazole core) plus unrelated small
# molecules, so scaffold counts, similarity and cliffs all have something real
# to find.
#
# Deliberately drug-like and not, say, a set of substituted benzenes: ECFP4
# Tanimoto rises with molecular size, because one substituent changes a smaller
# fraction of the bits on a big molecule than on a small one. A series built from
# two-ring fragments scores about 0.55 between neighbours and would sit below
# `_CLIFF_SIMILARITY` no matter how congeneric it obviously is -- which would
# make this file test the fixture's size rather than the cliff logic.
_SERIES = [
    "Cc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)F",
    "CCc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)F",
    "Clc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)F",
    "Cc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(C)(=O)=O)C(F)(F)F",
    "Cc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)Cl",
    "COc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)F",
]
_OTHERS = ["CCO", "CCCO", "c1ccncc1", "C1CCCCC1", "CC(=O)O", "CCN"]


def _frame(*, targets: list[float] | None = None, splits: list[str] | None = None) -> pl.DataFrame:
    structures = _SERIES + _OTHERS
    return pl.DataFrame(
        {
            "smiles": structures,
            "value": targets or [float(i) for i in range(len(structures))],
            "split": splits or (["train"] * 8 + ["validation"] * 2 + ["test"] * 2),
        }
    )


def _profile(frame: pl.DataFrame, kind: TargetKind = TargetKind.NUMERIC):
    return build_profile(
        frame=frame,
        structure_column="smiles",
        target=TargetSpec(column="value", kind=kind),
        normalizer=RdkitStructureNormalizer(),
    )


def test_train_and_test_histograms_share_their_edges() -> None:
    """Two histograms on independently chosen bins cannot be overlaid, and
    overlaying them is the whole point of carrying both."""
    profile = _profile(_frame())
    assert profile.target_distribution is not None
    histogram = profile.target_distribution.histogram
    assert len(histogram.edges) == len(histogram.train) + 1
    assert len(histogram.train) == len(histogram.test)
    for descriptor in profile.descriptors:
        assert len(descriptor.histogram.edges) == len(descriptor.histogram.train) + 1


def test_every_compound_is_counted_exactly_once_across_partitions() -> None:
    profile = _profile(_frame())
    assert sum(profile.partition_counts.values()) == profile.compounds


def test_a_constant_descriptor_correlates_to_none_not_zero() -> None:
    """`None` means unmeasurable; 0.0 means measured and unrelated. A constant
    target has no rank order, and reporting that as 0.0 would invite a reader to
    conclude the descriptor was checked and found irrelevant."""
    profile = _profile(_frame(targets=[1.0] * 12))
    assert all(d.target_correlation is None for d in profile.descriptors)
    assert profile.best_descriptor is None


def test_a_dominating_descriptor_is_surfaced_as_the_best_one() -> None:
    """The point of the correlation column: if one trivial property explains the
    target, then beating the mandatory baseline proves much less than it looks
    like it does, and this is the number that says so."""
    frame = _frame()
    # Set the target *to* molecular weight. Its own correlation must then be
    # perfect, and it must be the descriptor reported as the best.
    weights = RdkitStructureNormalizer().descriptors([str(s) for s in frame["smiles"].to_list()])[
        "molecular_weight"
    ]
    profile = _profile(frame.with_columns(pl.Series("value", weights)))

    best = next(d for d in profile.descriptors if d.name == "molecular_weight")
    assert best.target_correlation == pytest.approx(1.0)
    assert profile.best_descriptor == "molecular_weight"


def test_scaffold_overlap_separates_a_scaffold_split_from_a_random_one() -> None:
    """The integrity check on the split itself. A scaffold split that leaves
    scaffolds on both sides did not do what its name claims, and this is the
    number that makes the claim checkable instead of trusted."""
    # Random-shaped: the sulfonamide series straddles train and test.
    straddling = _frame(splits=["train", "test"] * 6)
    assert _profile(straddling).scaffolds.cross_split_scaffolds > 0

    # Scaffold-shaped: the whole series is in train, everything else in test.
    separated = _frame(splits=["train"] * 6 + ["test"] * 6)
    assert _profile(separated).scaffolds.cross_split_scaffolds == 0


def test_cumulative_coverage_is_monotonic_and_reaches_one() -> None:
    coverage = _profile(_frame()).scaffolds.cumulative_coverage
    assert coverage == sorted(coverage)
    assert coverage[-1] == pytest.approx(1.0)


def test_similarity_is_none_when_there_is_nothing_to_compare() -> None:
    """Never an all-zero histogram, which would read as 'every test compound is
    confirmed unlike anything trained on' rather than 'unmeasurable'."""
    assert _profile(_frame(splits=["train"] * 12)).similarity is None


def test_a_binary_target_reports_balance_per_partition_and_no_distribution() -> None:
    """Per partition because a split can leave a balanced dataset with a wildly
    unbalanced test set, which is what destabilises MCC."""
    frame = _frame(targets=[1.0] * 6 + [0.0] * 6)
    profile = _profile(frame, kind=TargetKind.BINARY)
    assert profile.target_distribution is None
    assert {b.split for b in profile.class_balance} <= {"train", "validation", "test"}
    assert sum(b.positive + b.negative for b in profile.class_balance) == profile.compounds


def test_activity_cliffs_rank_the_largest_disagreement_first() -> None:
    """Near-identical structures the assay disagrees about are the compounds a
    model reliably gets wrong; the biggest disagreement is the one to show."""
    targets = [1.0] * 12
    targets[1] = 9.0  # one analogue of the shared core is wildly more active
    profile = _profile(_frame(targets=targets))
    assert profile.activity_cliffs, "a congeneric series with a 8-unit jump has a cliff"
    top = profile.activity_cliffs[0]
    assert top.delta == pytest.approx(8.0)
    assert top.similarity >= 0.75
    assert profile.activity_cliffs == sorted(
        profile.activity_cliffs, key=lambda c: c.delta, reverse=True
    )


def test_identical_measurements_are_not_reported_as_cliffs() -> None:
    """Near-identical structures that agree are the ordinary case; including
    them would bury the handful that disagree."""
    assert _profile(_frame(targets=[3.0] * 12)).activity_cliffs == []


def test_a_small_dataset_does_not_claim_to_have_been_subsampled() -> None:
    assert _profile(_frame()).cliffs_sampled_from is None


def test_the_profile_survives_a_json_round_trip() -> None:
    """The cache blob is written with `profile_to_dict` and read back with
    `profile_from_dict`; a field that survives one and not the other would
    silently degrade every cached profile to a recompute."""
    profile = _profile(_frame())
    assert profile_from_dict(profile_to_dict(profile)) == profile


def test_acyclic_compounds_do_not_read_as_one_leaking_scaffold() -> None:
    """Regression: `murcko_scaffold` returns "" for every acyclic molecule, so
    grouping the cross-split check by the raw scaffold string lumps all of them
    into one family that spans every partition -- and reports a correctly-behaved
    scaffold split as leaking.

    `assign_split._scaffold_labels` deliberately keys each acyclic molecule by
    its own structure for exactly this reason, and this check has to key them the
    same way. Caught on the real BBBP dataset, where it claimed 95 compounds
    leaked across a split that had done nothing wrong.
    """
    acyclic = ["CCO", "CCCO", "CCCCO", "CCN", "CCCN", "CCCCN"]
    frame = pl.DataFrame(
        {
            "smiles": acyclic,
            "value": [float(i) for i in range(len(acyclic))],
            # Every partition holds acyclic molecules, and no two are the same.
            "split": ["train", "train", "train", "test", "test", "test"],
        }
    )
    profile = _profile(frame)
    assert profile.scaffolds.cross_split_scaffolds == 0
    assert profile.scaffolds.cross_split_compounds == 0
    # They are still one readable group in the display counts, which is a
    # genuinely useful fact about a dataset and a different question.
    assert profile.scaffolds.top[0].smiles == ""
    assert profile.scaffolds.top[0].count == len(acyclic)
