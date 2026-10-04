"""Small deterministic training frames shared by the engine tests."""

from __future__ import annotations

import polars as pl


def imbalanced_frame() -> pl.DataFrame:
    """300 rows: alcohols inactive, amines active (about 20%), deterministic split."""
    alcohols = [f"{'C' * n}O" for n in range(1, 121)]
    amines = [f"{'C' * n}N" for n in range(1, 31)]
    smiles = (alcohols + amines) * 2
    labels = ([0] * len(alcohols) + [1] * len(amines)) * 2
    split = ["train", "validation", "test", "train", "train"] * (len(smiles) // 5)
    return pl.DataFrame({"smiles": smiles, "y": labels, "split": split})


def two_binary_targets_frame() -> pl.DataFrame:
    """`imbalanced_frame`'s molecules with two labels: `a` is 1 for amines, `b` for odd
    chain lengths. Validation holds at least 10 actives and 10 inactives of each."""
    chain = pl.col("smiles").str.count_matches("C")
    return imbalanced_frame().select(
        "smiles",
        pl.col("smiles").str.ends_with("N").cast(pl.Int64).alias("a"),
        (chain % 2).cast(pl.Int64).alias("b"),
        "split",
    )
