import polars as pl
import pytest

from daikonstudio.application.data.assign_split import assign_split
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.data.target import RESERVED_TARGET_COLUMNS
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer

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
    assert "RANDOM" in excinfo.value.message


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
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        SplitSpec(strategy=SplitStrategy.RANDOM, seed=1, fractions=(0.5, 0.3, 0.3))


def test_split_spec_rejects_a_negative_fraction():
    with pytest.raises(ValueError, match="non-negative"):
        SplitSpec(strategy=SplitStrategy.RANDOM, seed=1, fractions=(1.1, -0.1, 0.0))
