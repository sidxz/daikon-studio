"""ESM-2's masked-marginal log-likelihood ratio.

Two tiers, the same split as `test_esm2_xgboost.py`. The arithmetic that turns
per-position log-probabilities into a per-variant score is pure and runs everywhere --
it is what breaks if the sign convention flips or a multiple mutant stops summing. The
forward pass is skipped unless the 2.6 GB checkpoint is already cached, because the API
tier installs without the gpu extra and CI does not download it.

The sign convention is the one the variant-effect literature uses and it is worth
stating once: LLR = log p(mutant) - log p(wild type), so a substitution the model
considers worse than the residue it replaced is NEGATIVE. Flipping it would invert every
correlation downstream while still producing plausible numbers.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from daikonstudio.infrastructure.protein.likelihood import _llr_from_logprobs, esm2_llr

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
_POS17_V = "MAKVQLSRLSGEQLLDEV"  # L17V
_DOUBLE = "MAKVQLSRLAGEQLLDEV"  # S9A + L17V

# Position 9's wild type is S, position 17's is L. Log-probabilities, not probabilities.
_LOGPROBS = {
    9: {"S": -0.1, "A": -2.1, "V": -3.0},
    17: {"L": -0.2, "V": -1.2},
}


def test_llr_is_zero_for_the_wild_type() -> None:
    """The wild type mutates nothing, so there is no ratio to take."""
    assert _llr_from_logprobs([_PARENT], _PARENT, _LOGPROBS)[:, 0] == pytest.approx([0.0])


def test_llr_is_the_log_ratio_at_the_mutated_site() -> None:
    """Negative, because the model prefers the serine it replaced. A positive number
    here would mean the sign convention had flipped."""
    assert _llr_from_logprobs([_POS9_A], _PARENT, _LOGPROBS)[:, 0] == pytest.approx([-2.0])


def test_llr_sums_over_every_mutated_site() -> None:
    result = _llr_from_logprobs([_DOUBLE], _PARENT, _LOGPROBS)

    assert result[:, 0] == pytest.approx([-3.0])  # (-2.1 + 0.1) + (-1.2 + 0.2)


def test_one_row_per_sequence_in_input_order() -> None:
    result = _llr_from_logprobs([_POS17_V, _PARENT, _POS9_A], _PARENT, _LOGPROBS)

    assert result.shape == (3, 1)
    assert result[:, 0] == pytest.approx([-1.0, 0.0, -2.0])


def test_a_residue_the_model_cannot_score_is_skipped_not_guessed() -> None:
    """An unscored residue contributes 0, which is log p(ratio) = 1 -- "no evidence",
    not "equally good". Guessing a value here would put a fabricated number into a
    feature column that a Scorecard reports as measured."""
    exotic = _PARENT[:9] + "X" + _PARENT[10:]

    assert _llr_from_logprobs([exotic], _PARENT, _LOGPROBS)[:, 0] == pytest.approx([0.0])


@needs_weights
def test_real_substitutions_mostly_score_below_their_wild_type() -> None:
    """The property that makes this feature worth a forward pass: on a real sequence
    ESM-2 should usually prefer the residue evolution put there. Not every substitution
    -- some are genuinely neutral or better -- so this asserts the majority, which is
    what a rank correlation against fitness actually rests on."""
    parent = "MAKVQLSRLSGEQLLDELMAKVQLSRLSGEQLLDEL"
    variants = [parent[:i] + "P" + parent[i + 1 :] for i in range(5, 15)]

    scores = esm2_llr([parent, *variants])

    assert scores.shape == (len(variants) + 1, 1)
    assert scores[0, 0] == pytest.approx(0.0), "the parent is its own reference"
    assert (scores[1:, 0] < 0).sum() >= 7, scores[1:, 0]


# --- the user path -----------------------------------------------------------------
#
# The feature existing is not the feature being reachable. These two assert the whole
# chain a person actually travels: the condition appears on the engine's manifest, so the
# picker renders it, and switching it on changes which featurizer the fit uses. A key
# typo between the two would leave the checkbox visible and inert.


def test_the_engine_offers_the_likelihood_condition() -> None:
    from daikonstudio.infrastructure.engines.esm2_xgboost import Esm2XGBoost

    conditions = {spec.key: spec for spec in Esm2XGBoost.manifest().conditions}

    assert "likelihood_feature" in conditions
    assert conditions["likelihood_feature"].default is False, "opt-in, not on by default"


def test_switching_it_on_changes_the_featurizer() -> None:
    from daikonstudio.infrastructure.engines._scoring import esm2_featurizer

    series = [_PARENT, _POS9_A, _POS17_V]

    assert esm2_featurizer({}, series)[0] == "esm2-650m"
    assert esm2_featurizer({"likelihood_feature": True}, series)[0] == "esm2-650m+llr"
