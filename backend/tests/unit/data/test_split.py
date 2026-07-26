import polars as pl
import pytest

from daikonstudio.application.data.assign_split import assign_split
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer

NORMALIZER = RdkitStructureNormalizer()

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


def test_scaffold_split_never_straddles_even_when_one_family_dominates():
    """A single scaffold owning 90% of the rows cannot fit an 80% train target -- the
    split must still keep the family whole rather than erroring or splitting it, even
    though the realized fractions end up nothing like 80/10/10 as a result. Known,
    accepted behaviour of the standard construction: see task-10-report.md."""
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
    result = assign_split(lopsided, "smiles", spec, NORMALIZER)
    benzenes = result.filter(pl.col("smiles").is_in(dominant))
    assert benzenes["split"].n_unique() == 1
    assert result["split"].null_count() == 0
    counts = dict(zip(*result["split"].value_counts().to_dict().values(), strict=True))
    assert sum(counts.values()) == 10
    assert max(counts.values()) == 9  # the dominant family, wherever it landed, is whole


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
