"""Choosing which protein model reads the sequences.

Three weight sets: stock ESM-2 650M, VESM2 650M laid over it, and ESM-2 150M. The first
two share an architecture and a width; the third does not, which is the whole reason
`Variant` carries a `dim` rather than this module carrying one constant.

The test that matters is `test_an_mlm_keyed_overlay_lands_nothing_on_the_embedding_model`.
VESM2 publishes an `EsmForMaskedLM` state dict and the embedding path loads an `EsmModel`,
so `load_state_dict(..., strict=False)` is not a tolerant load there -- it is a silent
no-op that returns stock ESM-2 from an engine the user selected VESM2 on. Measured on the
real file: 87 of 87 keys match the masked-LM class, 0 of 87 match the embedding class, and
81 of 81 match it once `esm.` is stripped.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.protein.embed import (
    DEFAULT_VARIANT,
    DIM,
    PROTEIN_MODEL_LABELS,
    PROTEIN_MODELS,
    VARIANTS,
    apply_overlay,
    site_dim,
    variant,
)

_WEIGHTS_DIR = Path(
    os.environ.get("STUDIO_PRETRAINED_WEIGHTS_DIR", "~/.cache/daikon-studio/weights")
).expanduser()

needs_weights = pytest.mark.skipif(
    not (_WEIGHTS_DIR / "hub" / "models--facebook--esm2_t33_650M_UR50D").exists(),
    reason="ESM-2 650M weights not cached; CI does not download 2.6 GB",
)
needs_vesm = pytest.mark.skipif(
    not (_WEIGHTS_DIR / "VESM_650M.pth").exists(),
    reason="VESM2 650M weights not cached; CI does not download 400 MB",
)


class _Stub:
    """Stands in for a loaded model: it knows its own parameter names and records what
    was laid over it. A real `EsmModel` would be 2.6 GB to answer a question about
    strings."""

    def __init__(self, names: list[str]) -> None:
        self._names = names
        self.loaded: dict[str, Any] = {}

    def state_dict(self) -> dict[str, Any]:
        return dict.fromkeys(self._names)

    def load_state_dict(self, state: dict[str, Any], strict: bool = True) -> None:
        self.loaded = state


_PUBLISHED = {
    "esm.embeddings.word_embeddings.weight": 1,
    "esm.encoder.layer.28.attention.self.query.weight": 2,
    "lm_head.dense.weight": 3,
}


def test_an_mlm_keyed_overlay_lands_nothing_on_the_embedding_model() -> None:
    """The failure the count exists to catch. `EsmModel`'s parameters are named
    `embeddings.*` and `encoder.*`; the published file names them `esm.embeddings.*`. With
    no stripping, not one tensor matches -- and `strict=False` says nothing about it."""
    model = _Stub(
        ["embeddings.word_embeddings.weight", "encoder.layer.28.attention.self.query.weight"]
    )

    landed = apply_overlay(model, dict(_PUBLISHED), for_masked_lm=True)

    assert landed == 0, "an MLM-keyed dict cannot match an EsmModel as published"
    assert model.loaded == {}


def test_stripping_the_prefix_is_what_makes_it_land() -> None:
    model = _Stub(
        ["embeddings.word_embeddings.weight", "encoder.layer.28.attention.self.query.weight"]
    )

    landed = apply_overlay(model, dict(_PUBLISHED), for_masked_lm=False)

    assert landed == 2
    assert set(model.loaded) == {
        "embeddings.word_embeddings.weight",
        "encoder.layer.28.attention.self.query.weight",
    }
    assert "lm_head.dense.weight" not in model.loaded, "the head is not part of EsmModel"


def test_a_masked_lm_takes_the_published_keys_unchanged() -> None:
    """The LLR path loads `EsmForMaskedLM`, which is what the file was keyed for, so
    nothing is stripped and the head lands too -- and the head is the half that produces
    a log-likelihood at all."""
    model = _Stub(list(_PUBLISHED))

    landed = apply_overlay(model, dict(_PUBLISHED), for_masked_lm=True)

    assert landed == 3
    assert "lm_head.dense.weight" in model.loaded


def test_a_partial_overlay_is_expected_not_tolerated() -> None:
    """`VESM_650M.pth` is 100M parameters against the base model's 652M: it overrides the
    embeddings, the top encoder layers and the head, and leaves the rest alone. So a
    count below the model's own parameter count is correct, and only zero is wrong."""
    model = _Stub([f"encoder.layer.{i}.weight" for i in range(33)])

    landed = apply_overlay(model, {"esm.encoder.layer.32.weight": 1}, for_masked_lm=False)

    assert landed == 1


# --- the roster ----------------------------------------------------------------------


def test_the_default_keeps_its_name() -> None:
    """Every artifact fitted before this choice existed names its featurizer
    `esm2-650m`, and `_load_bundle` has to still find it."""
    assert DEFAULT_VARIANT == "esm2-650m"
    assert PROTEIN_MODELS[0] == "esm2-650m"
    assert variant(DEFAULT_VARIANT).overlay is None


def test_every_variant_is_offered_with_a_label() -> None:
    assert len(PROTEIN_MODELS) == len(PROTEIN_MODEL_LABELS) == len(VARIANTS)
    assert all(label for label in PROTEIN_MODEL_LABELS)


def test_vesm3_is_not_on_the_roster() -> None:
    """It is ESM3-derived and carries EvolutionaryScale's non-commercial licence, and our
    runner images are public, which is distribution. Its absence is a decision."""
    assert not any("vesm3" in name for name in PROTEIN_MODELS)


def test_the_small_model_is_not_the_same_width() -> None:
    """The reason the width is per-variant rather than one module constant: a site block
    over ESM-2 150M is 1,922 columns, not 3,842."""
    assert variant("esm2-150m").dim == 640
    assert site_dim("esm2-150m") == 3 * 640 + 2
    assert site_dim() == 3 * DIM + 2


def test_an_unknown_model_is_refused_by_name() -> None:
    with pytest.raises(ValidationError, match="Unknown protein model"):
        variant("esm2-3b")


# --- the user path -------------------------------------------------------------------


def test_every_variant_has_all_four_representations_registered() -> None:
    """`_load_bundle` rebuilds a stored model's input from the bundle key alone, so every
    key an engine can compose has to resolve."""
    from daikonstudio.infrastructure.engines._scoring import _FEATURIZERS

    for model in PROTEIN_MODELS:
        for suffix in ("", "+site", "+llr", "+site+llr"):
            assert f"{model}{suffix}" in _FEATURIZERS, f"{model}{suffix}"


def test_the_engines_compose_the_key_from_the_chosen_model() -> None:
    from daikonstudio.infrastructure.engines._scoring import esm2_featurizer

    series = ["MAKVQLSRLSGEQLLDEL", "MAKVQLSRLAGEQLLDEL", "MAKVQLSRLSGEQLLDEV"]

    assert esm2_featurizer({}, series)[0] == "esm2-650m"
    assert esm2_featurizer({"protein_model": "vesm2-650m"}, series)[0] == "vesm2-650m"
    assert (
        esm2_featurizer({"protein_model": "esm2-150m", "site_features": True}, series)[0]
        == "esm2-150m+site"
    )
    assert (
        esm2_featurizer(
            {"protein_model": "vesm2-650m", "site_features": True, "likelihood_feature": True},
            series,
        )[0]
        == "vesm2-650m+site+llr"
    )


def test_every_esm2_engine_offers_the_choice() -> None:
    from daikonstudio.infrastructure.engines.esm2_gp import Esm2GP
    from daikonstudio.infrastructure.engines.esm2_xgboost import Esm2XGBoost

    for engine in (Esm2XGBoost, Esm2GP):
        conditions = {spec.key: spec for spec in engine.manifest().conditions}
        assert "protein_model" in conditions, engine.__name__
        assert conditions["protein_model"].default == DEFAULT_VARIANT
        assert tuple(conditions["protein_model"].options or ()) == PROTEIN_MODELS


# --- against the real weights --------------------------------------------------------


@needs_weights
@needs_vesm
def test_the_overlay_actually_changes_the_embeddings() -> None:
    """The end-to-end proof that selecting VESM2 does something. If the state dict landed
    nowhere -- the failure the unit tests above describe in the abstract -- these two
    calls would return identical arrays and the engine would report stock ESM-2 numbers
    under VESM2's name."""
    from daikonstudio.infrastructure.protein.embed import esm2_650m

    sequences = ["MAKVQLSRLSGEQLLDEL", "MAKVQLSRLAGEQLLDEL"]

    stock = esm2_650m(sequences, "esm2-650m")
    overlaid = esm2_650m(sequences, "vesm2-650m")

    assert stock.shape == overlaid.shape == (2, DIM)
    assert not np.allclose(stock, overlaid, atol=1e-3), "the overlay did not take effect"


@needs_weights
def test_the_small_model_embeds_at_its_own_width() -> None:
    """Asserted against the real forward pass rather than the dataclass, because the
    dataclass is exactly the kind of number that goes stale."""
    from daikonstudio.infrastructure.protein.embed import esm2_650m

    assert esm2_650m(["MAKVQLSRLSGEQLLDEL"], "esm2-150m").shape == (1, 640)
