"""The site-mean floor for variant series.

The behaviour under test is deliberately dull: predict the mean label of the training
variants that mutate the same position, and the global training mean when that position
was never trained on. What makes it worth an engine is that published supervised
variant-effect models frequently fail to beat it -- so a Scorecard without it on a
variant dataset is flattering its model.

The second test is the load-bearing one. Under a POSITION holdout every test position is
unseen by construction, so this engine collapses to a constant. That is correct, not
broken: it is the honest floor, and the gap between its random-split and
position-holdout numbers is exactly how much of a model's apparent skill was position
memorisation.
"""

from __future__ import annotations

import polars as pl
import pytest

from daikonstudio.application.engines.context import PredictContext, TrainContext
from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.engines.site_mean_fitness import SiteMeanFitness

# One parent, 18 residues, mutated at index 9 (S) and index 17 (L). Written out in full
# rather than built by string surgery: a reader has to be able to see that these are a
# single-parent series, which is the only regime this engine means anything in.
_PARENT = "MAKVQLSRLSGEQLLDEL"
_POS9_A = "MAKVQLSRLAGEQLLDEL"
_POS9_V = "MAKVQLSRLVGEQLLDEL"
_POS9_C = "MAKVQLSRLCGEQLLDEL"
_POS9_T = "MAKVQLSRLTGEQLLDEL"
_POS17_V = "MAKVQLSRLSGEQLLDEV"
_POS17_I = "MAKVQLSRLSGEQLLDEI"
_POS17_A = "MAKVQLSRLSGEQLLDEA"
_POS4_A = "MAKVALSRLSGEQLLDEL"


def _frame(labels: list[float]) -> pl.DataFrame:
    """Five train rows at two positions, three test rows: one at each trained position
    and one at a position the fit never saw.

    Three variants at position 9 against two at position 17 on purpose, so the global
    mean (0.56) is not the average of the two site means (0.5). With equal group sizes
    every expectation below would also be satisfied by a model that ignored position
    entirely and always returned the global mean.
    """
    return pl.DataFrame(
        {
            "sequence": [
                _POS9_A,
                _POS9_V,
                _POS9_C,
                _POS17_V,
                _POS17_I,
                _POS9_T,
                _POS17_A,
                _POS4_A,
            ],
            "y": labels,
            "split": ["train"] * 5 + ["test"] * 3,
        }
    )


def _train(labels: list[float], task: TaskType = TaskType.REGRESSION) -> bytes:
    return (
        SiteMeanFitness()
        .train(
            TrainContext(
                frame=_frame(labels),
                targets={"y": task},
                structure_column="sequence",
                conditions={},
                seed=42,
            )
        )
        .artifact
    )


def _predict(artifact: bytes, sequences: list[str]) -> list[float]:
    frame = SiteMeanFitness().predict(
        PredictContext(
            frame=pl.DataFrame({"sequence": sequences}),
            structure_column="sequence",
            artifact=artifact,
            conditions={},
            target_columns=("y",),
        )
    )
    return frame["value"].to_list()


# Train: position 9 averages 0.8, position 17 averages 0.2, everything averages 0.56.
_LABELS = [0.7, 0.9, 0.8, 0.1, 0.3, 0.0, 0.0, 0.0]
_GLOBAL = 0.56


def test_predicts_the_training_mean_at_the_same_position() -> None:
    artifact = _train(_LABELS)

    assert _predict(artifact, [_POS9_T, _POS17_A]) == pytest.approx([0.8, 0.2])


def test_falls_back_to_the_global_mean_at_an_unseen_position() -> None:
    """The POSITION-holdout case, and the wild type, which mutates nothing at all."""
    artifact = _train(_LABELS)

    assert _predict(artifact, [_POS4_A, _PARENT]) == pytest.approx([_GLOBAL, _GLOBAL])


def test_averages_the_positions_a_multiple_mutant_touches() -> None:
    artifact = _train(_LABELS)
    double = _POS9_A[:17] + "V"  # mutated at both 9 and 17

    # (0.8 + 0.2) / 2, which is NOT the global mean -- see `_frame`.
    assert _predict(artifact, [double]) == pytest.approx([0.5])


def test_refuses_unaligned_sequences() -> None:
    """ "Position 9" is not the same site in two sequences of different length, so a
    length mismatch is a different dataset shape rather than a harder fit."""
    ragged = _frame(_LABELS)["sequence"].to_list()
    ragged[1] += "GG"
    frame = _frame(_LABELS).with_columns(pl.Series("sequence", ragged))

    with pytest.raises(ValidationError, match="same length"):
        SiteMeanFitness().train(
            TrainContext(
                frame=frame,
                targets={"y": TaskType.REGRESSION},
                structure_column="sequence",
                conditions={},
                seed=42,
            )
        )


def test_classification_predicts_the_positive_fraction_at_the_position() -> None:
    """Site mean generalizes: for a binary endpoint it is the share of actives at the
    position, which is a probability and needs no rescaling."""
    artifact = _train([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], TaskType.BINARY_CLASSIFICATION)

    # position 9 is two of three active; position 17 is two of two.
    assert _predict(artifact, [_POS9_T, _POS17_A]) == pytest.approx([2 / 3, 1.0])


def test_reports_metrics_for_the_target_it_trained_on() -> None:
    result = SiteMeanFitness().train(
        TrainContext(
            frame=_frame(_LABELS),
            targets={"y": TaskType.REGRESSION},
            structure_column="sequence",
            conditions={},
            seed=42,
        )
    )

    assert set(result.metrics) == {"y"}
    assert "rmse" in result.metrics["y"]


def test_is_not_the_mandatory_baseline() -> None:
    """Exactly one engine may be flagged, and that is still the descriptor forest. This
    is an additional floor a user opts into, not a replacement for the required one."""
    assert SiteMeanFitness().manifest().is_baseline is False
