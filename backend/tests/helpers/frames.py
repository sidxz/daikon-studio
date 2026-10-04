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
