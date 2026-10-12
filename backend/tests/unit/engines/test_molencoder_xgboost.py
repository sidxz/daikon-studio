"""The MolEncoder engine's contract, on whatever device this machine has.

Two tiers, like `test_esm2_xgboost.py`. The manifest, the pin and the over-length
refusal need no checkpoint; the forward passes are skipped unless the 60 MB of weights
are already cached, because the default-lane worker and the API tier deliberately
install without the gpu extra and CI does not download them.

Nothing here pins the device. `protein.embed.torch_and_device` resolves CUDA, then MPS,
then CPU, and all three must satisfy the same contract.
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from daikonstudio.application.engines.context import PredictContext, TrainContext
from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.domain.data.structure_kind import StructureKind
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.chem import molencoder
from daikonstudio.infrastructure.engines.molencoder_xgboost import MolEncoderXGBoost

_WEIGHTS_DIR = Path(
    os.environ.get("STUDIO_PRETRAINED_WEIGHTS_DIR", "~/.cache/daikon-studio/weights")
).expanduser()
_SNAPSHOT = _WEIGHTS_DIR / "hub" / "models--fabikru--MolEncoder"

needs_weights = pytest.mark.skipif(
    not _SNAPSHOT.exists(),
    reason="MolEncoder weights not cached; CI does not download 60 MB",
)

#: Forty small molecules -- enough rows for a tree fit to have something to split on,
#: short enough that embedding them is a second or two.
_SMILES = [
    *(f"{'C' * n}O" for n in range(1, 9)),
    *(f"{'C' * n}N" for n in range(1, 9)),
    *(f"{'C' * n}C(=O)O" for n in range(1, 9)),
    *(f"{'C' * n}c1ccccc1" for n in range(1, 9)),
    *(f"{'C' * n}c1ccncc1" for n in range(1, 9)),
]
_SPLITS = ["train"] * 24 + ["validation"] * 8 + ["test"] * 8


def _frame(targets: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"smiles": _SMILES, "y": targets, "split": _SPLITS})


def test_the_manifest_declares_the_gpu_lane_and_reads_molecules() -> None:
    """`lane="gpu"` because this imports torch, and molecules because the featurizer
    tokenizes a SMILES string -- handed sequences it would still return a matrix of the
    right shape and the Scorecard would still render, on numbers that mean nothing."""
    manifest = MolEncoderXGBoost.manifest()

    assert manifest.id == "molencoder-xgboost"
    assert manifest.lane == "gpu"
    assert manifest.structure_kinds == (StructureKind.MOLECULE,)
    assert manifest.is_baseline is False


def test_the_engine_is_registered() -> None:
    from daikonstudio.infrastructure.engines.registry import default_registry

    assert "molencoder-xgboost" in {m.id for m in default_registry().manifests()}


def test_the_checkpoint_revision_is_pinned_to_a_commit_not_a_branch() -> None:
    """A published Protocol's predictions must not change because the Hub repository
    did. `6e443fbd` in the spec is the short form of this commit."""
    assert re.fullmatch(r"[0-9a-f]{40}", molencoder.REVISION), molencoder.REVISION
    assert molencoder.REVISION.startswith("6e443fbd")


def test_an_empty_partition_needs_no_forward_pass() -> None:
    """The width is a property of the checkpoint, so an empty partition must not require
    the gpu extra to be installed -- otherwise this passes here and fails in CI."""
    empty = molencoder.embed([])

    assert empty.shape == (0, molencoder.DIM)
    assert empty.dtype == np.float32


def test_a_structure_over_the_context_limit_is_refused_not_truncated() -> None:
    """Truncation would drop the tail of a molecule without saying so, and a peptide
    SMILES is almost all tail. A hand-built mask, so this runs without the checkpoint:
    the tokenizer is character-level, so a 600-character SMILES really does cost 602
    tokens.
    """
    torch = pytest.importorskip("torch")

    short, long = "CCO", "C" * (molencoder.MAX_TOKENS + 1)
    mask = torch.tensor([[1] * 5 + [0] * (len(long) - 5), [1] * len(long)])

    # At the limit exactly, nothing is refused.
    molencoder._refuse_over_length(torch.tensor([[1] * molencoder.MAX_TOKENS]), [short])

    with pytest.raises(ValidationError) as caught:
        molencoder._refuse_over_length(mask, [short, long])

    message = str(caught.value)
    assert f"at most {molencoder.MAX_TOKENS} tokens" in message
    # Counts the rows that are long and names the longest, as MoLFormer's guard does.
    assert "1 of these structures" in message
    assert long[:60] in message
    assert "fingerprint or descriptor engine" in message


@needs_weights
def test_embeddings_are_the_right_shape_and_reproducible() -> None:
    first = molencoder.embed(_SMILES[:8])
    second = molencoder.embed(list(_SMILES[:8]))

    assert first.shape == (8, molencoder.DIM)
    assert first.dtype == np.float32
    # Reproducible inference is load-bearing: a reloaded artifact must reproduce the
    # numbers its own Scorecard reported.
    np.testing.assert_allclose(first, second, rtol=0, atol=0)
    # Distinct molecules must not collapse onto one vector, or the representation is
    # useless whatever head sits on it.
    assert not np.allclose(first[0], first[1])


@needs_weights
def test_the_real_tokenizer_refuses_a_structure_past_the_context() -> None:
    """The guard above, reached through the tokenizer rather than a hand-built mask."""
    with pytest.raises(ValidationError, match="at most 502 tokens"):
        molencoder.embed(["CCO", "C" * 600])


@needs_weights
def test_regression_trains_predicts_and_round_trips() -> None:
    engine = MolEncoderXGBoost()
    result = engine.train(
        TrainContext(
            frame=_frame([float(i) for i in range(40)]),
            targets={"y": TaskType.REGRESSION},
            structure_column="smiles",
            conditions={"n_estimators": 20, "max_depth": 2},
            seed=7,
        )
    )

    assert sorted(result.metrics["y"]) == ["mae", "r2", "rmse", "spearman"]
    assert result.validation_metrics is not None
    # The bundle names its representation, which is the only thing that lets
    # `_predict_with_tree_ensemble` rebuild the input from bytes alone.
    assert pickle.loads(result.artifact)["featurizer"] == "molencoder"

    predictions = engine.predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": _SMILES[:6]}),
            structure_column="smiles",
            artifact=result.artifact,
            conditions={},
            target_columns=("y",),
        )
    )

    assert predictions.height == 6
    assert predictions.schema["value"] == pl.Float64
    # XGBoost has no ensemble spread to report, so the column is null rather than
    # fabricated -- and typed, or this engine's output would not concatenate with
    # another's.
    assert predictions.schema["uncertainty"] == pl.Float64
    assert predictions["uncertainty"].null_count() == 6
    assert predictions["value"].n_unique() > 1


@needs_weights
def test_classification_reports_the_shared_metric_vocabulary_and_never_accuracy() -> None:
    result = MolEncoderXGBoost().train(
        TrainContext(
            frame=_frame([float(i % 2) for i in range(40)]),
            targets={"y": TaskType.BINARY_CLASSIFICATION},
            structure_column="smiles",
            conditions={"n_estimators": 20, "max_depth": 2, "positive_weighting": "balanced"},
            seed=7,
        )
    )

    assert sorted(result.metrics["y"]) == ["auprc", "auroc", "balanced_accuracy", "mcc"]
    assert "accuracy" not in result.metrics["y"]
