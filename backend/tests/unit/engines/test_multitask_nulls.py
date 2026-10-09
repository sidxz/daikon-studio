"""Two engines declare `supports_multitask` and so bypass `FanOut`: they receive the
sparse frame whole. Each must either handle the nulls or refuse them -- never train on
them and return numbers, which is the one outcome with no visible symptom.

MoLFormer refuses: its loss masks nothing, so a null target would be arithmetic on a
missing value. Chemprop handles them, because chemprop v2 masks NaN targets in its loss
upstream -- behaviour this project depends on and does not own.
"""

from __future__ import annotations

import polars as pl
import pytest

from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.application.engines.registry import EngineRegistry
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.engines.chemprop_dmpnn import ChempropDMPNN
from daikonstudio.infrastructure.engines.molformer_xl import MolformerXL


def _sparse_context() -> object:
    from daikonstudio.application.engines.context import TrainContext

    frame = pl.DataFrame(
        {
            "smiles": ["CCO", "c1ccccc1", "CCN", "CCC"] * 8,
            "a": [1.0, 0.0, None, 1.0] * 8,
            "b": [0.0, 1.0, 1.0, None] * 8,
            "split": ["train", "train", "validation", "test"] * 8,
        }
    )
    return TrainContext(
        frame=frame,
        targets={"a": TaskType.BINARY_CLASSIFICATION, "b": TaskType.BINARY_CLASSIFICATION},
        structure_column="smiles",
        conditions={},
        seed=42,
    )


def test_molformer_refuses_a_sparse_dataset() -> None:
    """Named target, named count, and a pointer to the engine that can do the job --
    refused before the 1.1-billion-parameter checkpoint is even loaded, so this is a
    data-shape decision rather than a failed fit."""
    with pytest.raises(ValidationError) as caught:
        MolformerXL().train(_sparse_context())  # type: ignore[arg-type]

    message = str(caught.value)
    assert "'a'" in message
    assert "does not support" in message
    assert "Chemprop" in message


def test_every_multitask_engine_handles_or_refuses_nulls() -> None:
    """The mask lives in `FanOut`, which means it is invisible: an engine that declares
    `supports_multitask` opts out of it silently. This test is the whole defence. A new
    multitask engine must choose a branch; doing neither fails here rather than in a
    scorecard a scientist is reading."""
    registry = EngineRegistry({"chemprop-dmpnn": ChempropDMPNN(), "molformer-xl": MolformerXL()})
    multitask = [m.id for m in registry.manifests() if m.supports_multitask]

    assert set(multitask) == {"chemprop-dmpnn", "molformer-xl"}, (
        "a new multitask engine appeared; it must handle or refuse nulls, and this "
        "test is where it says which"
    )


# Fast enough to run in the unit suite; the shapes matter here, not convergence.
_FAST = {"epochs": 2, "depth": 2, "message_hidden_dim": 64, "batch_size": 8}


def test_chemprop_trains_through_unmeasured_targets() -> None:
    """The other branch, and the one that pins behaviour this project does not own:
    chemprop v2 masks NaN targets in its loss upstream, which is the entire reason
    Chemprop keeps the full sparse frame while every fanned-out engine gets a filtered
    one. If a version bump ever changes that, this test is the only thing that says so
    -- the symptom otherwise is a model quietly fitted to missing values."""
    from daikonstudio.application.engines.context import TrainContext

    # Each target is blank on rows the other is measured on, and each keeps both classes
    # among its *labelled* test rows -- otherwise its metrics come back undefined for a
    # reason that has nothing to do with the masking under test. Target `a` is blank on
    # one test row and `b` on another, which is the realistic shape.
    frame = pl.DataFrame(
        {
            "smiles": [
                "CCO",
                "CCN",
                "CCC",
                "CCCC",
                "CCCCC",
                "CCCCCC",
                "CCCCCCC",
                "CC(C)O",
                "CC(C)CO",
                "CCOC",
                "c1ccccc1",
                "Cc1ccccc1",
                "CCc1ccccc1",
                "c1ccncc1",
                "Cc1ccncc1",
                "CCOCC",
            ],
            # ten train, two validation, four test -- the last four of each row below
            "a": [
                1.0,
                0.0,
                None,
                1.0,
                0.0,
                1.0,
                None,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                None,
                1.0,
            ],
            "b": [
                None,
                1.0,
                1.0,
                0.0,
                None,
                0.0,
                1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                None,
                1.0,
                1.0,
            ],
            "split": ["train"] * 10 + ["validation"] * 2 + ["test"] * 4,
        }
    )
    ctx = TrainContext(
        frame=frame,
        targets={"a": TaskType.BINARY_CLASSIFICATION, "b": TaskType.BINARY_CLASSIFICATION},
        structure_column="smiles",
        conditions=_FAST,
        seed=13,
    )

    result = ChempropDMPNN().train(ctx)

    # Both targets scored, and neither score is NaN -- a loss that consumed the nulls
    # would propagate NaN into the weights and out through every metric.
    assert set(result.metrics) == {"a", "b"}
    for column, scores in result.metrics.items():
        assert scores, f"{column} produced no metrics"
        assert not all(value != value for value in scores.values()), (
            f"every metric for {column} is NaN: the loss consumed the unmeasured rows"
        )
