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
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.protein.embed import (
    DEFAULT_VARIANT,
    DIM,
    MAX_RESIDUES,
    PROTEIN_MODEL_LABELS,
    PROTEIN_MODELS,
    VARIANTS,
    _clean,
    aligned_parent,
    apply_overlay,
    likelihood_protocol,
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
needs_esmc = pytest.mark.skipif(
    not (_WEIGHTS_DIR / "hub" / "models--biohub--ESMC-600M").exists(),
    reason="ESM C 600M weights not cached; CI does not download 2.3 GB",
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


# --- ESM C -----------------------------------------------------------------------------


def test_esm_c_is_pinned_to_a_commit() -> None:
    """A mirror republished next week is a different set of weights under the same name,
    and the one on the roster is the one that was checked."""
    assert re.fullmatch(r"[0-9a-f]{40}", variant("esmc-600m").revision or "")
    assert variant("esmc-600m").checkpoint == "biohub/ESMC-600M"


def test_esm_c_carries_its_own_width_cap_and_layer_names() -> None:
    """All three differ from ESM-2's, and all three are read from the variant rather
    than from a module constant -- a LoRA fit that used ESM-2's layer names here would
    attach to nothing, and `_lora.attach` would refuse."""
    chosen = variant("esmc-600m")

    assert chosen.dim == 1152
    assert chosen.max_residues == 2048
    assert chosen.target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    assert variant(DEFAULT_VARIANT).max_residues == MAX_RESIDUES == 1022
    assert variant(DEFAULT_VARIANT).target_modules == ("query", "key", "value", "dense")


def test_the_residue_cap_is_the_chosen_models_and_truncation_is_counted() -> None:
    """Review item 5. The silent-loss rule is unchanged -- what changes is whose ceiling
    it is measured against. Truncating an ESM C run at ESM-2's 1,022 would throw away
    half a long protein, and truncating ESM-2 at 2,048 would hand it positions it has no
    embedding for."""
    long_one = ("M" * 2100,)

    cleaned, truncated, _unrepresentable = _clean(long_one, variant("esmc-600m").max_residues)
    assert truncated == 1, "over-length is counted, never silent"
    assert len(cleaned[0]) == 2048

    cleaned, truncated, _unrepresentable = _clean(long_one, variant(DEFAULT_VARIANT).max_residues)
    assert truncated == 1
    assert len(cleaned[0]) == 1022


def test_a_sequence_inside_esm_cs_context_is_not_refused_under_esm_2s() -> None:
    """`aligned_parent` guards the position-scored features, and its ceiling is the
    chosen model's too: a 1,500-residue protein is inside ESM C and outside ESM-2, and
    refusing it under the wrong number would read as a limit of the data."""
    series = ["M" * 1500, "A" + "M" * 1499]

    assert aligned_parent(series, None, "esmc-600m")[1] == "M" * 1500
    with pytest.raises(ValidationError, match="residue context"):
        aligned_parent(series, None, DEFAULT_VARIANT)


def test_the_likelihood_feature_is_refused_for_esm_c_rather_than_guessed() -> None:
    """The VESM2 lesson, applied before it costs anything. How a log-likelihood ratio is
    read off a checkpoint is a property of how that checkpoint was trained; scoring VESM2
    by ESM-2's protocol cost 0.147 Spearman and read as a weakness of the weights. Nobody
    has read Biohub's scoring code, so there is no protocol to use and the feature says
    so instead of picking one."""
    from daikonstudio.infrastructure.protein.likelihood import esm2_llr

    assert variant("esmc-600m").likelihood_scoring is None
    assert likelihood_protocol(DEFAULT_VARIANT) == "masked"
    assert likelihood_protocol("vesm2-650m") == "wt"

    with pytest.raises(ValidationError, match="not yet available"):
        likelihood_protocol("esmc-600m")
    # No weights load on either path: both refuse before any forward pass.
    with pytest.raises(ValidationError, match="not yet available"):
        esm2_llr(["MAKVQLSRLSGEQLLDEL"], None, "esmc-600m")


def test_the_engine_refuses_the_llr_toggle_before_it_embeds_anything() -> None:
    """The user path, which is the one that matters: the toggle is refused where the
    featurizer is resolved, not minutes later inside the forward pass."""
    from daikonstudio.infrastructure.engines._scoring import esm2_featurizer

    series = ["MAKVQLSRLSGEQLLDEL", "MAKVQLSRLAGEQLLDEL"]

    assert esm2_featurizer({"protein_model": "esmc-600m"}, series)[0] == "esmc-600m"
    with pytest.raises(ValidationError, match="not yet available"):
        esm2_featurizer({"protein_model": "esmc-600m", "likelihood_feature": True}, series)


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


@needs_esmc
def test_esm_c_embeds_at_its_own_width() -> None:
    """1,152 against ESM-2's 1,280, asserted on the real forward pass and not on the
    dataclass -- and the forward pass is also the only proof that `EsmcModel` loads
    through plain `AutoModel`, with no `trust_remote_code` and no `esm` package."""
    from daikonstudio.infrastructure.protein.embed import esm2_650m

    assert esm2_650m(["MAKVQLSRLSGEQLLDEL"], "esmc-600m").shape == (1, 1152)


@needs_esmc
def test_esm_c_maps_one_token_per_residue() -> None:
    """What `residue_token_offset` and every site feature are indexed by. A tokenizer
    that merged two residues into one token would shift every position after it and read
    a neighbouring residue while still returning a plausible number."""
    from daikonstudio.infrastructure.protein.embed import _loaded, residue_token_offset

    sequence = "MKTAYIAKQR"
    tokenizer, _model, _device = _loaded("esmc-600m")

    ids, offset = residue_token_offset(tokenizer, sequence)

    assert len(ids) == len(sequence) + 2, "one token per residue plus two special tokens"
    assert offset == 1


def test_prose_drops_the_pickers_parenthetical() -> None:
    """A variant's name has to read correctly inside a sentence, not just in the picker."""
    assert variant("esmc-600m").label == "ESM C 600M (faster)"
    assert variant("esmc-600m").prose == "ESM C 600M"
    # No parenthetical to drop: unchanged, so every message site can use prose.
    assert variant("esm2-650m").prose == variant("esm2-650m").label
