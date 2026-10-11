"""Site features: embeddings that say *where* a mutation is.

Two tiers, the same split as `test_protein_likelihood.py`. The parent guard and the
featurizer wiring are pure and run everywhere; the forward pass is skipped unless the
2.6 GB checkpoint is already cached, because the API tier installs without the gpu extra
and CI does not download it.

The sharp test in the weighted tier is `test_the_delta_is_measured_against_the_parent_at_
the_rows_own_site`. Every other assertion here would still pass if the delta block were
referenced to the wrong residue -- the numbers would stay the right shape, the right
magnitude and the wrong thing, which is the failure mode this whole feature invites.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.protein.embed import (
    DIM,
    MAX_RESIDUES,
    SITE_DIM,
    aligned_parent,
    esm2_with_sites,
)

_WEIGHTS_DIR = Path(
    os.environ.get("STUDIO_PRETRAINED_WEIGHTS_DIR", "~/.cache/daikon-studio/weights")
).expanduser()
_SNAPSHOT = _WEIGHTS_DIR / "hub" / "models--facebook--esm2_t33_650M_UR50D"

needs_weights = pytest.mark.skipif(
    not _SNAPSHOT.exists(),
    reason="ESM-2 650M weights not cached; CI does not download 2.6 GB",
)

_PARENT = "MAKVQLSRLSGEQLLDEL"
_POS9_A = "MAKVQLSRLAGEQLLDEL"  # S9A
_POS9_C = "MAKVQLSRLCGEQLLDEL"  # S9C
_POS17_V = "MAKVQLSRLSGEQLLDEV"  # L17V
_DOUBLE = "MAKVQLSRLAGEQLLDEV"  # S9A + L17V


# --- the parent guard, shared with the likelihood feature ---------------------------


def test_the_parent_is_the_modal_residue() -> None:
    normalized, parent = aligned_parent([_PARENT, _POS9_A, _POS9_C, _POS17_V])

    assert parent == _PARENT
    assert normalized == [_PARENT, _POS9_A, _POS9_C, _POS17_V]


def test_lowercase_rows_are_not_read_as_mutations() -> None:
    """The reason normalization happens before the consensus and not after. Un-normalized,
    `a` and `A` are different residues, so a lowercase row reads as a variant at every
    position -- of a parent that exists in no well."""
    normalized, parent = aligned_parent([_PARENT, _PARENT.lower(), " MAKVQ LSRLSGEQLLDEL "])

    assert parent == _PARENT
    assert normalized == [_PARENT] * 3


def test_a_ragged_series_is_refused() -> None:
    """A position-scored feature has no meaning without an alignment, and `consensus`
    would otherwise fail on `zip(strict=True)` with nothing a user could act on."""
    with pytest.raises(ValidationError, match="same length"):
        aligned_parent([_PARENT, _PARENT[:-1]])


def test_a_series_past_the_context_window_is_refused() -> None:
    """Truncating would move every position past the cut, so the positions the feature
    is indexed by would no longer be the positions the model saw."""
    with pytest.raises(ValidationError, match="residue context"):
        aligned_parent(["M" * (MAX_RESIDUES + 1)] * 2)


def test_the_likelihood_feature_shares_the_guard() -> None:
    """One definition of the parent for both position-scored features: if these two
    disagreed about the wild type, the LLR and the site delta would describe different
    substitutions and both would still return plausible numbers."""
    from daikonstudio.infrastructure.protein.likelihood import esm2_llr

    with pytest.raises(ValidationError, match="same length"):
        esm2_llr([_PARENT, _PARENT[:-1]])


def test_no_sequences_is_an_empty_block_of_the_right_width() -> None:
    assert esm2_with_sites([]).shape == (0, SITE_DIM)
    assert SITE_DIM == 3 * DIM + 2


# --- the forward pass ---------------------------------------------------------------


@needs_weights
def test_the_block_is_pooled_then_site_then_delta_then_scalars() -> None:
    features = esm2_with_sites([_PARENT, _POS9_A, _POS17_V])

    assert features.shape == (3, SITE_DIM)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


@needs_weights
def test_the_pooled_half_is_the_embedding_the_plain_featurizer_produces() -> None:
    """The site block pools the same hidden states it gathers residues from, so it must
    agree with `esm2_650m` -- `allclose` and not `array_equal`, because a different batch
    composition changes the reduction order by about 1e-06 per dimension."""
    from daikonstudio.infrastructure.protein.embed import esm2_650m

    sequences = [_PARENT, _POS9_A, _POS17_V]

    assert np.allclose(esm2_with_sites(sequences)[:, :DIM], esm2_650m(sequences), atol=1e-4)


@needs_weights
def test_the_wild_type_gets_zeros_and_a_mutation_count_of_zero() -> None:
    """Zeros rather than a fabricated vector for a row that mutates nothing, with the
    count beside them so a tree can learn the site block means nothing on that row."""
    wild_type = esm2_with_sites([_PARENT, _POS9_A, _POS17_V])[0]

    assert (wild_type[DIM : 3 * DIM] == 0).all()
    assert wild_type[3 * DIM] == 0.0
    assert wild_type[3 * DIM + 1] == 0.0
    assert wild_type[:DIM].any(), "the pooled half is still a real embedding"


@needs_weights
def test_the_site_block_moves_with_the_mutated_position() -> None:
    """The whole point of the feature: a point mutation moves a 286-residue mean by about
    1/286, so two variants at different sites are nearly identical in the pooled half and
    must not be in the site half."""
    features = esm2_with_sites([_PARENT, _POS9_A, _POS17_V])
    at_9, at_17 = features[1], features[2]

    assert not np.allclose(at_9[DIM : 2 * DIM], at_17[DIM : 2 * DIM], atol=1e-2)
    assert at_9[DIM : 2 * DIM].any() and at_9[2 * DIM : 3 * DIM].any()


@needs_weights
def test_the_delta_is_measured_against_the_parent_at_the_rows_own_site() -> None:
    """`site - delta` recovers the parent's state at whichever position that row mutates.
    Two variants of the same position must recover the same vector and a variant of a
    different position must not -- which is what fails if the delta is referenced to a
    fixed position, to the pooled parent, or to an off-by-one token.
    """
    features = esm2_with_sites([_PARENT, _POS9_A, _POS9_C, _POS17_V])
    reference = features[:, DIM : 2 * DIM] - features[:, 2 * DIM : 3 * DIM]

    assert np.allclose(reference[1], reference[2], atol=1e-3), "both mutate position 9"
    assert not np.allclose(reference[1], reference[3], atol=1e-2), "9 is not 17"


@needs_weights
def test_the_scalars_are_the_mean_relative_position_and_the_mutation_count() -> None:
    """A multiple mutant averages over its sites, matching how the LLR sums over them.
    13/18 is neither 9/18 nor 17/18, so this fixture fails if the mean is dropped.

    Two copies of the parent, which is not padding. With one each of the parent and the
    double mutant, position 17 holds two Ls and two Vs, and `consensus` breaks a tie by
    residue letter -- so V becomes the wild type, every row reads as a variant of a parent
    that is in no well, and the scalars below come out plausible and wrong. A real DMS set
    has the wild-type residue dominating every column; a four-row fixture does not.
    """
    series = [_PARENT, _PARENT, _POS9_A, _POS17_V, _DOUBLE]
    assert aligned_parent(series)[1] == _PARENT, "the fixture has to recover the parent"

    features = esm2_with_sites(series)

    assert features[2, 3 * DIM] == pytest.approx(9 / 18)
    assert features[3, 3 * DIM] == pytest.approx(17 / 18)
    assert features[4, 3 * DIM] == pytest.approx(13 / 18)
    assert list(features[:, 3 * DIM + 1]) == [0.0, 0.0, 1.0, 1.0, 2.0]


# --- the user path -------------------------------------------------------------------
#
# The feature existing is not the feature being reachable: the condition has to appear on
# the manifest for the picker to render it, and switching it on has to change which
# featurizer the fit uses. A key typo between the two leaves the checkbox visible and
# inert.


def test_the_engine_offers_the_site_condition() -> None:
    from daikonstudio.infrastructure.engines.esm2_xgboost import Esm2XGBoost

    conditions = {spec.key: spec for spec in Esm2XGBoost.manifest().conditions}

    assert "site_features" in conditions
    assert conditions["site_features"].default is False, "opt-in, not on by default"


def test_the_parent_is_settled_from_the_whole_frame_and_recorded() -> None:
    """The fit records the reference its features were scored against.

    Without this, the parent is whatever the list in hand implies, and the two paths that
    featurize never hold the whole series: a fit featurizes train, validation and test
    separately, and `predict` featurizes the frame it is handed.
    """
    from daikonstudio.infrastructure.engines._scoring import bundle_features, esm2_featurizer

    whole_frame = [_PARENT, _POS9_A, _POS17_V]
    bundle_key, _featurizer, parent = esm2_featurizer({"site_features": True}, whole_frame)

    assert parent == _PARENT
    assert bundle_features(bundle_key, parent)["parent"] == _PARENT
    # A plain embedding describes a sequence, not a substitution, so it has no reference
    # to record and its bundle must not grow a key.
    assert esm2_featurizer({}, whole_frame)[2] is None
    assert "parent" not in bundle_features("esm2-650m", None)


@needs_weights
def test_one_variant_predicted_alone_is_not_its_own_parent() -> None:
    """The bug this plumbing exists for, asserted end to end.

    A single-row predict frame has a consensus of itself, so an inferred parent makes the
    row read as zero mutations: its site block comes out zeros and its LLR comes out 0,
    from a model fitted on real values. Nothing in the output says so. Bound to the
    recorded parent, the lone row features exactly as it did inside the fit.
    """
    from daikonstudio.infrastructure.engines._scoring import esm2_featurizer

    series = [_PARENT, _POS9_A, _POS9_C, _POS17_V]
    _bundle_key, featurizer, parent = esm2_featurizer({"site_features": True}, series)

    in_series = featurizer(series)[1]
    alone = featurizer([_POS9_A])[0]
    inferred_alone = esm2_with_sites([_POS9_A])[0]

    assert np.allclose(in_series, alone, atol=1e-4), "same row, same features"
    assert alone[3 * DIM + 1] == 1.0, "one mutation, against the recorded parent"
    assert inferred_alone[3 * DIM + 1] == 0.0, "the defect: its own parent, no mutations"
    assert parent == _PARENT


def test_predicting_a_different_protein_is_refused_not_guessed() -> None:
    """A model carrying a 18-residue parent cannot score a 20-residue variant: there is
    no alignment between them, so every position would be the wrong position."""
    with pytest.raises(ValidationError, match="recorded when it was trained"):
        aligned_parent([_PARENT + "GG"], parent=_PARENT)


def test_the_two_toggles_compose_without_moving_the_older_keys() -> None:
    """Four combinations, and the two that existed before keep their exact spellings --
    an artifact fitted before the site block existed names its featurizer in its bundle,
    and `_load_bundle` has to still find it."""
    from daikonstudio.infrastructure.engines._scoring import _FEATURIZERS, esm2_featurizer

    series = [_PARENT, _POS9_A, _POS17_V]

    assert esm2_featurizer({}, series)[0] == "esm2-650m"
    assert esm2_featurizer({"likelihood_feature": True}, series)[0] == "esm2-650m+llr"
    assert esm2_featurizer({"site_features": True}, series)[0] == "esm2-650m+site"
    both = esm2_featurizer({"site_features": True, "likelihood_feature": True}, series)
    assert both[0] == "esm2-650m+site+llr"
    assert all(
        key in _FEATURIZERS
        for key in ("esm2-650m", "esm2-650m+llr", "esm2-650m+site", "esm2-650m+site+llr")
    )
