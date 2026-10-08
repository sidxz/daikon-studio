import itertools
import shutil

import polars as pl
import pytest

from daikonstudio.application.data.assign_split import assign_split
from daikonstudio.domain.data.split import (
    SplitSpec,
    SplitStrategy,
    is_replicable,
    split_from_dict,
    split_to_dict,
)
from daikonstudio.domain.data.target import RESERVED_TARGET_COLUMNS
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer
from daikonstudio.infrastructure.protein.cluster import Mmseqs2Clusterer

NORMALIZER = RdkitStructureNormalizer()


def test_assign_split_only_ever_injects_a_reserved_column_name():
    """Whole-branch review follow-up: the reserved set that C1 introduced
    went stale within the same commit that created it (a `_row_number`
    column added elsewhere collided with it, undetected). This asserts
    against the *real* output of `assign_split` -- not a hand-maintained
    mirror of what the function is believed to inject -- so a future column
    added here without updating `RESERVED_TARGET_COLUMNS` fails this test
    rather than shipping a silent-overwrite bug.
    """
    frame = pl.DataFrame({"smiles": ["CCO", "CCN"]})
    result = assign_split(
        frame, "smiles", SplitSpec(strategy=SplitStrategy.RANDOM, seed=1), NORMALIZER
    )
    injected = set(result.columns) - set(frame.columns)
    assert injected  # the assertion below is meaningless if this ever becomes empty
    assert injected <= RESERVED_TARGET_COLUMNS


SMILES = [
    "CCO",
    "CCCO",
    "CCCCO",
    "c1ccccc1",
    "Cc1ccccc1",
    "CCc1ccccc1",
    "CCN",
    "CCCN",
    "c1ccncc1",
    "Cc1ccncc1",
]


def frame() -> pl.DataFrame:
    return pl.DataFrame({"smiles": SMILES})


def test_random_split_is_reproducible_from_the_seed():
    spec = SplitSpec(strategy=SplitStrategy.RANDOM, seed=7)
    first = assign_split(frame(), "smiles", spec, NORMALIZER)["split"].to_list()
    second = assign_split(frame(), "smiles", spec, NORMALIZER)["split"].to_list()
    assert first == second


def test_random_split_respects_the_fractions():
    spec = SplitSpec(strategy=SplitStrategy.RANDOM, seed=7, fractions=(0.8, 0.1, 0.1))
    counts = assign_split(frame(), "smiles", spec, NORMALIZER)["split"].value_counts().to_dict()
    assert dict(zip(counts["split"], counts["count"], strict=True))["train"] == 8


def test_scaffold_split_keeps_a_scaffold_within_one_partition():
    """Benzene, toluene and ethylbenzene share a scaffold and must not straddle splits."""
    spec = SplitSpec(strategy=SplitStrategy.SCAFFOLD, seed=7)
    result = assign_split(frame(), "smiles", spec, NORMALIZER)
    benzenes = result.filter(pl.col("smiles").is_in(["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1"]))
    assert benzenes["split"].n_unique() == 1


def test_every_row_receives_a_partition():
    spec = SplitSpec(strategy=SplitStrategy.SCAFFOLD, seed=7)
    result = assign_split(frame(), "smiles", spec, NORMALIZER)
    assert result["split"].null_count() == 0
    assert set(result["split"].unique()) <= {"train", "validation", "test"}


def test_scaffold_split_is_deterministic_across_calls():
    """Same (frame, seed) must yield the same assignment -- no hidden global RNG state."""
    spec = SplitSpec(strategy=SplitStrategy.SCAFFOLD, seed=7)
    first = assign_split(frame(), "smiles", spec, NORMALIZER)["split"].to_list()
    second = assign_split(frame(), "smiles", spec, NORMALIZER)["split"].to_list()
    assert first == second


def test_acyclic_molecules_are_not_forced_into_one_family():
    """Murcko returns "" for every acyclic molecule, but they aren't one scaffold family --
    lumping all five into a single artificial group would let coincidence (no ring) dictate
    a partition, so each is its own singleton group and free to land independently."""
    spec = SplitSpec(strategy=SplitStrategy.SCAFFOLD, seed=7)
    result = assign_split(frame(), "smiles", spec, NORMALIZER)
    acyclic = result.filter(pl.col("smiles").is_in(["CCO", "CCCO", "CCCCO", "CCN", "CCCN"]))
    assert acyclic["split"].n_unique() > 1


def test_scaffold_split_raises_when_a_dominant_family_leaves_a_partition_empty():
    """A single scaffold owning 90% of the rows cannot fit an 80% train target. The
    greedy fill still never splits the family -- but it does leave `test` completely
    empty, which is an untrainable/unevaluable result, not a lopsided-but-usable one.
    That must fail loudly rather than come back as a normal-looking frame."""
    dominant = [
        "c1ccccc1",
        "Cc1ccccc1",
        "CCc1ccccc1",
        "CCCc1ccccc1",
        "CCCCc1ccccc1",
        "CCCCCc1ccccc1",
        "CCCCCCc1ccccc1",
        "Fc1ccccc1",
        "Clc1ccccc1",
    ]
    lopsided = pl.DataFrame({"smiles": [*dominant, "C1CCNCC1"]})
    spec = SplitSpec(strategy=SplitStrategy.SCAFFOLD, seed=7)
    with pytest.raises(ValidationError) as excinfo:
        assign_split(lopsided, "smiles", spec, NORMALIZER)
    assert "test" in excinfo.value.message
    assert "90%" in excinfo.value.message
    assert "c1ccccc1" in excinfo.value.message
    assert "random split" in excinfo.value.message


def test_scaffold_split_raises_when_one_scaffold_covers_every_row():
    """The degenerate case: every row shares one scaffold, so no partition split is
    possible at all without straddling it. Train ends up empty."""
    frame_one_family = pl.DataFrame(
        {"smiles": ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1"]}
    )
    spec = SplitSpec(strategy=SplitStrategy.SCAFFOLD, seed=7)
    with pytest.raises(ValidationError) as excinfo:
        assign_split(frame_one_family, "smiles", spec, NORMALIZER)
    assert "train" in excinfo.value.message
    assert "100%" in excinfo.value.message


def test_scaffold_split_raises_on_ordinary_evenly_sized_families():
    """Not a degenerate input: four scaffold families of five rows each (a completely
    ordinary dataset shape) still overshoots train and skips validation entirely --
    the greedy fill's failure mode isn't limited to extreme dominance."""
    rings = ["c1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1"]
    prefixes = ["", "C", "CC", "CCC", "CCCC"]
    rows = [prefix + ring for ring in rings for prefix in prefixes]
    frame_even_families = pl.DataFrame({"smiles": rows})
    spec = SplitSpec(strategy=SplitStrategy.SCAFFOLD, seed=7)
    with pytest.raises(ValidationError) as excinfo:
        assign_split(frame_even_families, "smiles", spec, NORMALIZER)
    assert "validation" in excinfo.value.message
    assert "25%" in excinfo.value.message


def test_scaffold_split_acyclic_singleton_tie_break_depends_on_seed_not_file_order():
    """Acyclic rows tie at group size 1; which of them is held out for validation/test
    must come from the seed, not from wherever they happened to sit in the uploaded
    file -- otherwise a CSV sorted by potency would leak a systematic bias into what
    looks like a random tie-break. The realized counts (8/1/1) stay stable across
    seeds; only which specific row lands where should change."""
    acyclic = ["CCO", "CCCO", "CCCCO", "CCN", "CCCN"]
    patterns = set()
    for seed in range(1, 6):
        spec = SplitSpec(strategy=SplitStrategy.SCAFFOLD, seed=seed)
        result = assign_split(frame(), "smiles", spec, NORMALIZER)
        labels = dict(zip(result["smiles"].to_list(), result["split"].to_list(), strict=True))
        patterns.add(tuple(labels[s] for s in acyclic))
        benzenes = result.filter(pl.col("smiles").is_in(["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1"]))
        assert benzenes["split"].n_unique() == 1
    assert len(patterns) > 1


# Twelve real, ring-bearing molecules, each a distinct Murcko scaffold -- every one
# of them a "singleton" scaffold family, none of them acyclic. This is the fixture
# whose absence let the seed be inert on real chemical data: an earlier fix only
# seeded the tie-break for empty-scaffold (acyclic/invalid) rows, so on data where
# every molecule has a ring -- the common case, since most drug-like molecules do --
# ties still broke by file order, and the module's own docstring wrongly claimed
# immunity to it.
RING_BEARING_SMILES = [
    "c1ccccc1",  # benzene
    "c1ccncc1",  # pyridine
    "c1ccoc1",  # furan
    "c1ccsc1",  # thiophene
    "C1CCCC1",  # cyclopentane
    "C1CCCCC1",  # cyclohexane
    "C1CCNC1",  # pyrrolidine
    "C1CCOC1",  # tetrahydrofuran
    "c1ccc2ccccc2c1",  # naphthalene
    "C1CC1",  # cyclopropane
    "C1CCC1",  # cyclobutane
    "C1CCNCC1",  # piperidine
]


def test_scaffold_split_seed_matters_when_every_molecule_has_a_ring():
    """The generalized fix: on a fixture where nothing is acyclic, different seeds
    must still produce different assignments. Before the fix, this printed the same
    holdout for every seed tried, because only empty-scaffold ties were seeded."""
    ring_frame = pl.DataFrame({"smiles": RING_BEARING_SMILES})
    patterns = set()
    for seed in (0, 1, 42, 999, 123456):
        spec = SplitSpec(strategy=SplitStrategy.SCAFFOLD, seed=seed)
        result = assign_split(ring_frame, "smiles", spec, NORMALIZER)
        assert result["split"].null_count() == 0
        patterns.add(tuple(result["split"].to_list()))
    assert len(patterns) > 1


def test_scaffold_split_holdout_is_order_invariant_when_every_molecule_has_a_ring():
    """A CSV sorted by potency or appended chronologically must not change which
    molecules get held out, for a fixed seed -- reversing the row order here must
    not change the holdout set."""
    spec = SplitSpec(strategy=SplitStrategy.SCAFFOLD, seed=7)
    forward = assign_split(
        pl.DataFrame({"smiles": RING_BEARING_SMILES}), "smiles", spec, NORMALIZER
    )
    backward = assign_split(
        pl.DataFrame({"smiles": list(reversed(RING_BEARING_SMILES))}), "smiles", spec, NORMALIZER
    )
    forward_holdout = set(forward.filter(pl.col("split") != "train")["smiles"].to_list())
    backward_holdout = set(backward.filter(pl.col("split") != "train")["smiles"].to_list())
    assert forward_holdout == backward_holdout


def test_tiny_frame_leaves_some_partitions_empty_but_never_crashes():
    """Two rows, a three-way split: pigeonhole guarantees an empty partition. Pinned,
    not left to chance -- the larger fraction (train) absorbs the rows."""
    spec = SplitSpec(strategy=SplitStrategy.RANDOM, seed=7)
    result = assign_split(pl.DataFrame({"smiles": ["CCO", "CCN"]}), "smiles", spec, NORMALIZER)
    counts = dict(zip(*result["split"].value_counts().to_dict().values(), strict=True))
    assert counts == {"train": 2}


def test_split_spec_rejects_fractions_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match=r"sum to 1\."):
        SplitSpec(strategy=SplitStrategy.RANDOM, seed=1, fractions=(0.5, 0.3, 0.3))


def test_split_spec_rejects_a_negative_fraction():
    with pytest.raises(ValueError, match="non-negative"):
        SplitSpec(strategy=SplitStrategy.RANDOM, seed=1, fractions=(1.1, -0.1, 0.0))


# --- Sequence strategies -------------------------------------------------------------
#
# IDENTITY and POSITION are the two protein regimes, and they are not interchangeable:
# on a single-parent variant series every sequence is homologous to every other, so
# IDENTITY correctly returns one cluster and cannot split it -- which is what POSITION
# exists for. Measured on real ProteinGym assays in `infrastructure/protein/cluster.py`.

PARENT = "MAKVQLSRLSGEQLLDELSRRFGGKVNVY"

#: A single mutant at 5, the double mutant at (5, 9), and a single mutant at 9. These
#: three are the leakage case: grouped by *position set* they would be three different
#: groups, so position 5 could be trained on and held out at the same time.
LINKED = ((5,), (5, 9), (9,))
VARIANT_POSITIONS = (*LINKED, (2,), (12,), (15,), (18,), (21,), (24,), (27,))


def variant(*positions: int) -> str:
    """The parent with a substitution at each 0-based position, never the parent residue."""
    residues = list(PARENT)
    for position in positions:
        residues[position] = "A" if residues[position] != "A" else "G"
    return "".join(residues)


def variant_frame() -> pl.DataFrame:
    """The wild type plus ten variants -- so the consensus is the parent at every
    position (no position is mutated in more than 2 of 11 rows)."""
    return pl.DataFrame({"sequence": [PARENT, *(variant(*p) for p in VARIANT_POSITIONS)]})


def mutated_positions(sequence: str) -> set[int]:
    return {i for i, (a, b) in enumerate(zip(sequence, PARENT, strict=True)) if a != b}


class FixedClusterer:
    """A `SequenceClusterer` returning ids handed to it.

    MMseqs2 is not what these tests are checking, and a unit test must not depend on a
    binary that is absent on the API tier, in CI and on most developer machines. The
    adapter's own real-data behaviour is gated on `shutil.which` below.
    """

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def cluster(self, sequences: list[str], *, min_identity: float, coverage: float) -> list[int]:
        assert len(sequences) == len(self._ids)
        assert (min_identity, coverage) == (0.3, 0.8)  # the thresholds that recover families
        return self._ids


def test_position_split_never_lets_one_position_reach_two_partitions():
    """The whole point of the strategy. If any residue position is mutated in both a
    training row and a test row, the model has already seen that site vary and the
    holdout measures nothing."""
    spec = SplitSpec(strategy=SplitStrategy.POSITION, seed=7)
    result = assign_split(variant_frame(), "sequence", spec, NORMALIZER)
    per_partition: dict[str, set[int]] = {}
    for sequence, label in zip(result["sequence"], result["split"], strict=True):
        per_partition.setdefault(label, set()).update(mutated_positions(sequence))
    assert len(per_partition) > 1  # a single partition would make the assertion vacuous
    for left, right in itertools.combinations(per_partition.values(), 2):
        assert not left & right


def test_position_split_merges_co_mutated_positions_into_one_group():
    """Grouping by position *set* is the obvious implementation and it leaks: the single
    mutant at 5 and the double mutant at (5, 9) would be different groups. Union-find
    merges 5 and 9 into one component, so all three linked rows share a partition."""
    spec = SplitSpec(strategy=SplitStrategy.POSITION, seed=7)
    result = assign_split(variant_frame(), "sequence", spec, NORMALIZER)
    linked = result.filter(pl.col("sequence").is_in([variant(*p) for p in LINKED]))
    assert linked.height == len(LINKED)
    assert linked["split"].n_unique() == 1


def test_position_split_is_deterministic_and_order_invariant():
    """Both inferred quantities -- the consensus residue and a component's id -- are
    resolved by taking a minimum, not by first appearance, so re-sorting the uploaded
    rows cannot move the split for a fixed seed."""
    spec = SplitSpec(strategy=SplitStrategy.POSITION, seed=7)
    rows = variant_frame()["sequence"].to_list()
    forward = assign_split(pl.DataFrame({"sequence": rows}), "sequence", spec, NORMALIZER)
    backward = assign_split(
        pl.DataFrame({"sequence": list(reversed(rows))}), "sequence", spec, NORMALIZER
    )
    labels = dict(zip(forward["sequence"], forward["split"], strict=True))
    assert dict(zip(backward["sequence"], backward["split"], strict=True)) == labels


def test_position_split_rejects_sequences_of_different_lengths():
    """ "Position 37" is not the same site in two sequences of different length, so an
    unaligned series would merge unrelated sites into one component and the holdout
    would be meaningless rather than merely lopsided."""
    ragged = pl.DataFrame({"sequence": [PARENT, PARENT + "A", PARENT[:-3]]})
    spec = SplitSpec(strategy=SplitStrategy.POSITION, seed=7)
    with pytest.raises(ValidationError) as excinfo:
        assign_split(ragged, "sequence", spec, NORMALIZER)
    assert "same length" in excinfo.value.message
    assert "identity split" in excinfo.value.message


def test_identity_split_keeps_a_homology_cluster_within_one_partition():
    spec = SplitSpec(strategy=SplitStrategy.IDENTITY, seed=7)
    ids = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6]
    sequences = pl.DataFrame({"sequence": [variant(i) for i in range(len(ids))]})
    result = assign_split(sequences, "sequence", spec, NORMALIZER, clusterer=FixedClusterer(ids))
    per_cluster: dict[int, set[str]] = {}
    for cluster, label in zip(ids, result["split"].to_list(), strict=True):
        per_cluster.setdefault(cluster, set()).add(label)
    assert all(len(labels) == 1 for labels in per_cluster.values())
    assert result["split"].n_unique() > 1  # otherwise the assertion above is vacuous


def test_identity_split_without_a_clusterer_raises():
    """The port is optional on the signature (every other strategy ignores it), so a
    caller that forgot to wire it must fail with something an operator can act on."""
    spec = SplitSpec(strategy=SplitStrategy.IDENTITY, seed=7)
    with pytest.raises(ValidationError) as excinfo:
        assign_split(variant_frame(), "sequence", spec, NORMALIZER)
    assert "sequence-clustering tool" in excinfo.value.message


def test_identity_split_on_one_parents_variants_points_at_the_position_strategy():
    """The single most likely way this strategy is misapplied: variants of one protein
    are all homologous, so MMseqs2 returns one cluster and no split is possible. The
    error is the only place a scientist learns which strategy they wanted instead, so
    it must not talk about Bemis-Murcko scaffolds and compounds."""
    spec = SplitSpec(strategy=SplitStrategy.IDENTITY, seed=7)
    frame_one_family = variant_frame()
    clusterer = FixedClusterer([0] * frame_one_family.height)
    with pytest.raises(ValidationError) as excinfo:
        assign_split(frame_one_family, "sequence", spec, NORMALIZER, clusterer=clusterer)
    message = excinfo.value.message
    assert "homologous" in message
    assert "position split" in message
    assert "100%" in message
    assert "scaffold" not in message


def test_every_split_strategy_is_dispatched_explicitly():
    """The root cause this build fixed: dispatch was `if RANDOM: ... else: scaffold`, so
    a member added to the enum silently got a scaffold split while the Dataset, the
    Scorecard and the UI all reported the strategy the scientist asked for. Adding a
    member now fails here -- `assign_split`'s `assert_never` raises for an unhandled
    one, and the equality below refuses to let this test fall behind the enum."""
    import itertools as _it

    _base = frame()
    predefined_frame = _base.with_columns(
        pl.Series(
            "split",
            list(_it.islice(_it.cycle(["train", "test", "validation"]), _base.height)),
        )
    )
    # Each entry is (frame, structure column, assign_split kwargs, SplitSpec kwargs).
    inputs: dict[SplitStrategy, tuple[pl.DataFrame, str, dict[str, object], dict[str, object]]] = {
        SplitStrategy.RANDOM: (frame(), "smiles", {}, {}),
        SplitStrategy.SCAFFOLD: (frame(), "smiles", {}, {}),
        SplitStrategy.IDENTITY: (
            variant_frame(),
            "sequence",
            {"clusterer": FixedClusterer(list(range(variant_frame().height)))},
            {},
        ),
        SplitStrategy.POSITION: (variant_frame(), "sequence", {}, {}),
        SplitStrategy.PREDEFINED: (predefined_frame, "smiles", {}, {"column": "split"}),
    }
    assert set(inputs) == set(SplitStrategy)
    for strategy, (rows, column, extra, spec_extra) in inputs.items():
        spec = SplitSpec(strategy=strategy, seed=7, **spec_extra)  # type: ignore[arg-type]
        result = assign_split(rows, column, spec, NORMALIZER, **extra)  # type: ignore[arg-type]
        assert result["split"].null_count() == 0, strategy
        assert set(result["split"].unique()) <= {"train", "validation", "test"}, strategy


@pytest.mark.skipif(
    shutil.which("mmseqs") is None,
    reason="MMseqs2 not installed; CI and the API tier do not ship the binary",
)
def test_identity_split_runs_end_to_end_against_the_real_clusterer():
    """The one test that exercises the adapter and the split together. It asserts the
    invariant against the clusterer's *own* output rather than an expected grouping, so
    it stays honest whatever MMseqs2 makes of a handful of short sequences."""
    sequences = [variant(i) for i in range(10)]
    clusterer = Mmseqs2Clusterer()
    ids = clusterer.cluster(sequences, min_identity=0.3, coverage=0.8)
    spec = SplitSpec(strategy=SplitStrategy.IDENTITY, seed=7)
    try:
        result = assign_split(
            pl.DataFrame({"sequence": sequences}),
            "sequence",
            spec,
            NORMALIZER,
            clusterer=clusterer,
        )
    except ValidationError as error:
        # One cluster covering all ten variants is the correct answer for a
        # single-parent series, and the refusal is the documented behaviour.
        assert len(set(ids)) == 1
        assert "position split" in error.message
        return
    per_cluster: dict[int, set[str]] = {}
    for cluster, label in zip(ids, result["split"].to_list(), strict=True):
        per_cluster.setdefault(cluster, set()).add(label)
    assert all(len(labels) == 1 for labels in per_cluster.values())


# --- PREDEFINED: the partitions the file declares ------------------------------------
#
# A published benchmark ships its own train/test assignment, and reproducing its number
# means using that assignment rather than one we computed. Everything below is about
# refusing to pretend: a spec that claims PREDEFINED without naming a column, or names a
# column while claiming a strategy that computes its own partitions, is not a split this
# app can honour.


def test_predefined_requires_a_column():
    with pytest.raises(ValueError, match="needs the name of the column"):
        SplitSpec(strategy=SplitStrategy.PREDEFINED, seed=42)


def test_other_strategies_refuse_a_column():
    with pytest.raises(ValueError, match="computes its own partitions"):
        SplitSpec(strategy=SplitStrategy.SCAFFOLD, seed=42, column="split")


def test_predefined_refuses_non_default_fractions():
    # Fractions mean nothing when the partitions are given, and an inert non-default
    # value would sit on the Dataset looking like it had done something.
    with pytest.raises(ValueError, match="do not apply"):
        SplitSpec(
            strategy=SplitStrategy.PREDEFINED,
            seed=42,
            column="split",
            fractions=(0.7, 0.2, 0.1),
        )


def test_predefined_round_trips_through_the_jsonb_shape():
    spec = SplitSpec(strategy=SplitStrategy.PREDEFINED, seed=42, column="Set")
    assert split_to_dict(spec)["column"] == "Set"
    assert split_from_dict(split_to_dict(spec)) == spec


def test_a_split_frozen_before_this_feature_still_loads():
    # Every split ever written lacks the key; a KeyError here would break loading
    # every existing dataset in the workspace.
    spec = split_from_dict({"strategy": "scaffold", "seed": 7, "fractions": [0.8, 0.1, 0.1]})
    assert spec.strategy is SplitStrategy.SCAFFOLD
    assert spec.column is None


class _NeverNormalizer:
    """PREDEFINED reads a column; it must never look at a structure."""

    def canonicalize(self, smiles):  # pragma: no cover - asserted not to run
        raise AssertionError("a predefined split must not canonicalize")

    def has_multiple_components(self, smiles):  # pragma: no cover
        raise AssertionError("a predefined split must not inspect components")

    def murcko_scaffold(self, smiles):  # pragma: no cover
        raise AssertionError("a predefined split must not scaffold")


def _predefined(values, column="split"):
    spec = SplitSpec(strategy=SplitStrategy.PREDEFINED, seed=42, column=column)
    rows = pl.DataFrame({"smiles": [f"C{'C' * i}O" for i in range(len(values))], column: values})
    return assign_split(rows, "smiles", spec, _NeverNormalizer())  # type: ignore[arg-type]


def test_predefined_reads_the_partitions_in_file_order():
    result = _predefined(["train", "test", "validation", "train"])
    assert result["split"].to_list() == ["train", "test", "validation", "train"]


def test_predefined_accepts_case_and_surrounding_whitespace():
    # Excel and hand-editing produce these constantly; rejecting them would be a
    # worse answer than reading them.
    result = _predefined(["Train ", "TEST", " Validation", "train"])
    assert result["split"].to_list() == ["train", "test", "validation", "train"]


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("training", "train"),
        ("valid", "validation"),
        ("val", "validation"),
        ("dev", "validation"),
        ("testing", "test"),
    ],
)
def test_predefined_accepts_common_synonyms(spelling, expected):
    result = _predefined(["train", "test", spelling])
    assert result["split"].to_list()[2] == expected


def test_predefined_rejects_an_unrecognised_value_naming_the_column_and_value():
    with pytest.raises(ValidationError) as caught:
        _predefined(["train", "holdout", "test"])
    message = caught.value.message
    assert "split" in message
    assert "holdout" in message


def test_predefined_rejects_numeric_fold_indices_and_says_what_it_wants():
    # TDC ships five numbered splits and QMAP five test sets, so a scientist will try
    # this. k-fold columns are out of scope, and the message has to say so.
    with pytest.raises(ValidationError) as caught:
        _predefined(["0", "1", "2"])
    assert "train" in caught.value.message


def test_predefined_rejects_a_file_with_no_training_rows():
    with pytest.raises(ValidationError, match="no training rows"):
        _predefined(["test", "test", "test"])


def test_predefined_rejects_a_file_with_no_test_rows():
    with pytest.raises(ValidationError, match="no test rows"):
        _predefined(["train", "train"])


def test_predefined_allows_a_two_partition_split():
    # MoleculeACE, Polaris and CardioTox are all train/test only. Carving a validation
    # partition out of train here would make our training set smaller than the one the
    # published number came from.
    result = _predefined(["train", "train", "test"])
    assert set(result["split"].to_list()) == {"train", "test"}


def test_predefined_reads_a_column_that_is_not_called_split():
    result = _predefined(["train", "test"], column="Set")
    assert result["split"].to_list() == ["train", "test"]
    # The source column is left alone; only the canonical "split" column is injected.
    assert result["Set"].to_list() == ["train", "test"]


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (SplitStrategy.RANDOM, True),
        (SplitStrategy.SCAFFOLD, True),
        (SplitStrategy.POSITION, True),
        (SplitStrategy.IDENTITY, False),
        (SplitStrategy.PREDEFINED, False),
    ],
)
def test_which_splits_can_be_reseeded(strategy: SplitStrategy, expected: bool) -> None:
    assert is_replicable(strategy) is expected


def test_every_strategy_has_an_answer() -> None:
    """A sixth strategy must choose a side here rather than inherit one: the budget,
    the worker's gate and the form all read this, so a strategy it has no answer for
    would be promised draws nobody takes."""
    for strategy in SplitStrategy:
        assert isinstance(is_replicable(strategy), bool)
