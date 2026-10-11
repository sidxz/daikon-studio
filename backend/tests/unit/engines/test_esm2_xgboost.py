"""The ESM-2 engine's contract, on whatever device this machine has.

Two tiers, on purpose. The pooling and batching tests below run everywhere, because they
are arithmetic and do not need a checkpoint -- they are what fails if the mean-pool mask
or the token-budget batching breaks. The end-to-end fit is skipped unless the 2.6 GB
checkpoint is already cached, for the same reason `test_molformer_xl.py` skips: the
default-lane worker and the API tier deliberately install without the gpu extra, and CI
does not download it.

Nothing here pins the device. `protein/embed.py:_loaded` resolves CUDA, then MPS, then
CPU, and all three must satisfy the same contract.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from daikonstudio.infrastructure.protein.embed import (
    MAX_RESIDUES,
    _batches,
    _clean,
    esm2_650m,
)

_WEIGHTS_DIR = Path(
    os.environ.get("STUDIO_PRETRAINED_WEIGHTS_DIR", "~/.cache/daikon-studio/weights")
).expanduser()
_SNAPSHOT = _WEIGHTS_DIR / "hub" / "models--facebook--esm2_t33_650M_UR50D"

needs_weights = pytest.mark.skipif(
    not _SNAPSHOT.exists(),
    reason="ESM-2 650M weights not cached; CI does not download 2.6 GB",
)

# Real M. tuberculosis-shaped toy input: short fragments standing in for variants of one
# parent, which is the regime this engine is for.
_SEQUENCES = [
    "MAKVQLSRLSGEQLLDEL",
    "MAKVQLSRLSGEQLLDEV",
    "MAKVQLSRLSGEQLLDEI",
    "MAKVQLSRLAGEQLLDEL",
    "MAKVQLSRLTGEQLLDEL",
    "MAKVQLSRLCGEQLLDEL",
    "MPKVQLSRLSGEQLLDEL",
    "MTKVQLSRLSGEQLLDEL",
]


def test_truncation_is_counted_not_silent() -> None:
    """The whole point of `_clean` returning a count: RpoB and EmbB exceed the cap, so a
    silent cut would quietly describe only part of the protein."""
    cleaned, truncated, _unrepresentable = _clean(("M" * (MAX_RESIDUES + 500), "MAKV"))
    assert truncated == 1
    assert len(cleaned[0]) == MAX_RESIDUES
    assert cleaned[1] == "MAKV"


def test_clean_normalizes_whitespace_gaps_and_case() -> None:
    cleaned, truncated, _unrepresentable = _clean(("ma kv\n", "MA-KV", "MA.KV"))
    assert cleaned == ["MAKV", "MAKV", "MAKV"]
    assert truncated == 0


def test_batches_cover_every_index_exactly_once() -> None:
    """A dropped or duplicated index would silently misalign embeddings with labels --
    the worst possible failure here, because it would still train and still score."""
    cleaned = ["M" * n for n in (10, 900, 20, 5, 1000, 3)]
    order = sorted(range(len(cleaned)), key=lambda i: len(cleaned[i]), reverse=True)
    batches = _batches(cleaned, order)
    flat = [i for batch in batches for i in batch]
    assert sorted(flat) == list(range(len(cleaned)))
    assert len(flat) == len(set(flat))


def test_batches_respect_the_token_budget() -> None:
    """One long sequence must not drag a wide batch along with it."""
    cleaned = ["M" * 1000] + ["M" * 1000 for _ in range(40)]
    order = list(range(len(cleaned)))
    batches = _batches(cleaned, order)
    assert len(batches) > 1
    for batch in batches:
        longest = max(len(cleaned[i]) for i in batch)
        assert len(batch) * (longest + 2) <= 16384 + (longest + 2)


@needs_weights
def test_embeddings_are_deterministic_and_the_right_shape() -> None:
    from daikonstudio.infrastructure.protein.embed import DIM, esm2_650m

    first = esm2_650m(_SEQUENCES)
    second = esm2_650m(list(_SEQUENCES))
    assert first.shape == (len(_SEQUENCES), DIM)
    assert first.dtype == np.float32
    # Reproducible inference is load-bearing: a reloaded artifact must reproduce the
    # numbers its own Scorecard reported.
    np.testing.assert_allclose(first, second, rtol=0, atol=0)
    # Distinct point mutations must not collapse to one vector, or the featurizer is
    # useless for the variant-effect regime this engine targets.
    assert not np.allclose(first[0], first[1])


@needs_weights
def test_padding_does_not_change_a_sequence_embedding() -> None:
    """Mean-pooling must mask padding and the start/end tokens. Batched with a much
    longer sequence, a short one would otherwise drift -- this is the test that fails if
    the pooling mask regresses."""
    from daikonstudio.infrastructure.protein.embed import esm2_650m

    alone = esm2_650m(["MAKVQLSRLSGEQLLDEL"])
    with_a_long_neighbour = esm2_650m(["MAKVQLSRLSGEQLLDEL", "M" * 600])
    np.testing.assert_allclose(alone[0], with_a_long_neighbour[0], rtol=1e-4, atol=1e-4)


@needs_weights
def test_regression_trains_predicts_and_round_trips() -> None:
    from daikonstudio.application.engines.context import PredictContext, TrainContext
    from daikonstudio.application.engines.manifest import TaskType
    from daikonstudio.infrastructure.engines.esm2_xgboost import Esm2XGBoost

    frame = pl.DataFrame(
        {
            "sequence": _SEQUENCES,
            "mic": [0.5, 2.0, 4.0, 0.6, 1.0, 8.0, 0.4, 0.7],
            "split": ["train"] * 4 + ["validation", "validation", "test", "test"],
        }
    )
    engine = Esm2XGBoost()
    result = engine.train(
        TrainContext(
            frame=frame,
            targets={"mic": TaskType.REGRESSION},
            structure_column="sequence",
            conditions={"n_estimators": 20, "max_depth": 2},
            seed=7,
        )
    )
    assert "mic" in result.metrics
    assert result.validation_metrics is not None
    assert result.artifact

    predictions = engine.predict(
        PredictContext(
            frame=pl.DataFrame({"sequence": _SEQUENCES[:3]}),
            structure_column="sequence",
            artifact=result.artifact,
            conditions={},
            target_columns=("mic",),
        )
    )
    assert predictions.height == 3
    assert set(predictions.columns) >= {"row_id", "value"}
    assert predictions["value"].null_count() == 0


# --- reproducibility, measured rather than assumed ---------------------------------


@needs_weights
def test_embedding_is_batch_composition_stable_only_to_float32_tolerance() -> None:
    """The guarantee this module actually has, pinned so nobody upgrades it in a comment.

    A batch pads to its longest member, so the same sequence beside different neighbours
    goes through differently shaped matmuls and the reduction order changes. `allclose`
    holds; `array_equal` does not, and PyTorch never promised it would.

    Asserting the inequality too, deliberately. If a future change DOES make this
    bitwise stable -- embedding the whole frame once and slicing is the way -- this test
    fails and whoever did it gets to delete the tolerance and the paragraph in
    `esm2_650m` that explains it, rather than leaving a stale caveat behind.
    """
    target = "MAKVQLSRLSGEQLLDELMAKVQLSRLSGEQLLDEL"
    neighbours = ["M" + "AKVQLSRLSGEQLLDEL" * 20] * 6

    alone = esm2_650m([target])[0]
    batched = esm2_650m([target, *neighbours])[0]
    reordered = esm2_650m([*neighbours, target])[-1]

    assert np.allclose(alone, batched, atol=1e-5)
    assert np.allclose(alone, reordered, atol=1e-5)
    assert not np.array_equal(alone, batched), (
        "bitwise stability would be better than the tolerance this pins -- see docstring"
    )


def test_a_silent_precision_downgrade_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """`DNNL_DEFAULT_FPMATH_MODE=BF16` appears in AWS's own recommended Graviton recipe,
    and it makes every "fp32" matmul lower precision without saying so. A deployment can
    acquire it without anyone choosing it, so the loader refuses rather than quietly
    reporting scores computed at a precision the artifact does not record."""
    from daikonstudio.domain.shared.errors import ValidationError
    from daikonstudio.infrastructure.protein.embed import torch_and_device

    monkeypatch.setenv("DNNL_DEFAULT_FPMATH_MODE", "BF16")

    with pytest.raises(ValidationError, match="lower precision"):
        torch_and_device()


def test_strict_and_unset_precision_are_both_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    from daikonstudio.infrastructure.protein.embed import torch_and_device

    monkeypatch.setenv("DNNL_DEFAULT_FPMATH_MODE", "strict")
    torch_and_device()  # does not raise

    monkeypatch.delenv("DNNL_DEFAULT_FPMATH_MODE")
    torch_and_device()


# --- residues ESM-2 cannot represent -----------------------------------------------


def test_unrepresentable_residues_are_counted_not_silent() -> None:
    """`*` is the one character that passes our sequence detection and still lands on
    ESM-2's `<unk>` token: `X`, `B`, `Z`, `U` and `O` are all real tokens it was trained
    with. Degrading one row rather than killing the fit is the right policy -- but it
    has to be visible, exactly as truncation is, or a nonsense variant is embedded as if
    it were an ordinary protein."""
    cleaned, truncated, unrepresentable = _clean(("MAKV*QLSR", "MAKVQLSR", "MXBZUO"))

    assert truncated == 0
    assert unrepresentable == 1, "only the stop codon is outside ESM-2's alphabet"
    assert cleaned[0] == "MAKV*QLSR", "counted, not rewritten -- the row still trains"


@needs_weights
def test_the_embeddable_alphabet_matches_the_real_tokenizer() -> None:
    """`_EMBEDDABLE` is hardcoded so `_clean` can run without loading 2.6 GB. This is
    what keeps it honest: every residue it claims is representable must be a real token,
    and the one it excludes must actually be `<unk>`."""
    from daikonstudio.infrastructure.protein.embed import _EMBEDDABLE, _loaded

    tokenizer, _model, _device = _loaded()

    for residue in sorted(_EMBEDDABLE):
        assert tokenizer.convert_tokens_to_ids(residue) != tokenizer.unk_token_id, residue
    assert tokenizer.convert_tokens_to_ids("*") == tokenizer.unk_token_id
