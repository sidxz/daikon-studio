"""The ESM-2 Gaussian process: the length scale, the prior mean, and the round trip.

Two tiers, as in `test_esm2_xgboost.py`. The length-scale heuristic and the prior-mean
arithmetic are pure and run everywhere; the end-to-end fit needs the 2.6 GB checkpoint
and is skipped without it.

The test worth reading is `test_the_length_scale_starts_from_the_data_not_from_one`. The
failure it guards is not a crash: at sklearn's default length scale of 1.0 every
off-diagonal entry of an RBF over standardized 1280-dimensional vectors underflows, the
GP learns nothing, and it returns its prior mean for every row with a straight face.
That is the shape of the published claim that ESM-2 embeddings "cannot be used by GPs".
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from daikonstudio.infrastructure.engines.esm2_gp import (
    Esm2GP,
    _kernel,
    _median_distance,
    _PriorMeanGP,
)

_WEIGHTS_DIR = Path(
    os.environ.get("STUDIO_PRETRAINED_WEIGHTS_DIR", "~/.cache/daikon-studio/weights")
).expanduser()
_SNAPSHOT = _WEIGHTS_DIR / "hub" / "models--facebook--esm2_t33_650M_UR50D"

needs_weights = pytest.mark.skipif(
    not _SNAPSHOT.exists(),
    reason="ESM-2 650M weights not cached; CI does not download 2.6 GB",
)

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


# --- the length scale ----------------------------------------------------------------


def test_the_length_scale_starts_from_the_data_not_from_one() -> None:
    """Standardized 1280-dimensional rows sit roughly sqrt(2 * 1280) ~ 50 apart, so a
    length scale of 1.0 -- sklearn's default -- puts exp(-50^2 / 2) in every off-diagonal
    entry, which is zero in float64. The kernel becomes the identity, the GP has no
    neighbours to borrow from, and every prediction is the prior mean."""
    rng = np.random.default_rng(0)
    scaled = rng.standard_normal((200, 1280))

    scale = _median_distance(scaled, seed=0)

    assert 30.0 < scale < 80.0, scale
    kernel = _kernel(scaled, seed=0)
    # The thing that actually matters: a pair at the typical distance is not zero.
    similarity = kernel(scaled[:2], scaled[2:4])
    assert similarity.max() > 1e-6, "the kernel has to see neighbours as neighbours"


def test_duplicate_rows_do_not_set_the_length_scale() -> None:
    """A single-parent series is full of near-duplicates, and a zero distance says
    nothing about the scale on which the function varies -- but it is the median if
    enough rows coincide."""
    row = np.ones((1, 8))
    scaled = np.vstack([np.repeat(row, 9, axis=0), row * 50.0])

    assert _median_distance(scaled, seed=0) > 1.0


def test_identical_rows_fall_back_rather_than_divide_by_nothing() -> None:
    """Every row the same leaves no positive distance to take a median of. 1.0 is a
    fallback, not a claim; the WhiteKernel absorbs what it cannot explain."""
    assert _median_distance(np.ones((5, 4)), seed=0) == 1.0


# --- the prior mean ------------------------------------------------------------------


class _ConstantGP:
    """Stands in for a fitted GP that has learned nothing, so the only thing left in the
    prediction is the prior mean this wrapper adds back."""

    def predict(self, x: np.ndarray, return_std: bool = False) -> object:
        zeros = np.zeros(x.shape[0])
        return (zeros, np.full(x.shape[0], 0.25)) if return_std else zeros


class _Identity:
    def transform(self, x: np.ndarray) -> np.ndarray:
        return x


def test_the_prior_mean_is_a_line_in_the_likelihood_column() -> None:
    """The LLR is the featurizer's last column. With a GP that predicts 0, the output is
    exactly `slope * llr + intercept`, which is what "prior mean" has to mean."""
    model = _PriorMeanGP(_ConstantGP(), _Identity(), prior=(2.0, 1.0))
    x = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.5]])

    assert list(model.predict(x)) == pytest.approx([-1.0, 2.0])


def test_the_likelihood_column_is_kept_out_of_the_kernel() -> None:
    """Kermut uses the LLR as the mean, not as a kernel input, and its ablations measure
    it there. This asserts the split: the scaler and GP see 1280 columns, not 1281."""
    seen: list[int] = []

    class _Recording(_ConstantGP):
        def predict(self, x: np.ndarray, return_std: bool = False) -> object:
            seen.append(x.shape[1])
            return super().predict(x, return_std)

    model = _PriorMeanGP(_Recording(), _Identity(), prior=(1.0, 0.0))
    model.predict(np.zeros((3, 1281)))

    assert seen == [1280]


def test_without_the_likelihood_every_column_is_a_kernel_input() -> None:
    """`prior=None` is the no-LLR case, and then nothing may be stripped -- taking the
    last column off anyway would silently drop an embedding dimension."""
    seen: list[int] = []

    class _Recording(_ConstantGP):
        def predict(self, x: np.ndarray, return_std: bool = False) -> object:
            seen.append(x.shape[1])
            return super().predict(x, return_std)

    model = _PriorMeanGP(_Recording(), _Identity(), prior=None)

    assert list(model.predict(np.zeros((2, 1280)))) == pytest.approx([0.0, 0.0])
    assert seen == [1280]


def test_an_empty_partition_predicts_nothing_rather_than_raising() -> None:
    """`_scored` always featurizes the test partition, and an empty one is legitimate: a
    sparse target in a fanned-out leg can have no measured test row, which
    `regression_metrics` reports as undefined. `StandardScaler.transform` raises on zero
    rows where the tree engines return an empty array, so without this the whole run
    fails on a sklearn message that names neither the engine nor the cause."""
    model = _PriorMeanGP(_ConstantGP(), _Identity(), prior=(1.0, 0.0))

    assert model.predict(np.zeros((0, 1281))).shape == (0,)
    value, deviation = model.predict(np.zeros((0, 1281)), return_std=True)
    assert value.shape == (0,) and deviation.shape == (0,)


def test_the_standard_deviation_is_not_shifted_by_the_prior() -> None:
    """Adding a mean moves the location, never the spread. A prior-shifted uncertainty
    would be reported in the Scorecard's uncertainty column as if it were measured."""
    model = _PriorMeanGP(_ConstantGP(), _Identity(), prior=(10.0, 5.0))

    _value, deviation = model.predict(np.array([[0.0, 3.0]]), return_std=True)

    assert list(deviation) == pytest.approx([0.25])


# --- the user path -------------------------------------------------------------------


def test_the_engine_is_regression_only_and_says_so() -> None:
    """Not a gap: `GaussianProcessClassifier` exposes no latent variance, so a
    classification branch would report the same boundary-distance proxy as every other
    engine, from a model that costs cubically more. `train_protocol` refuses by manifest
    and the picker filters on this."""
    from daikonstudio.application.engines.manifest import TaskType

    assert Esm2GP.manifest().tasks == (TaskType.REGRESSION,)


def test_the_engine_is_registered() -> None:
    from daikonstudio.infrastructure.engines.registry import default_registry

    assert default_registry().get("esm2-gp") is not None


def test_a_training_set_too_large_for_a_cubic_fit_is_refused() -> None:
    """Named rather than OOM-killed. The ceiling is the one property of this engine a
    scientist has to know before choosing it."""
    from daikonstudio.application.engines.context import TrainContext
    from daikonstudio.application.engines.manifest import TaskType
    from daikonstudio.domain.shared.errors import ValidationError

    frame = pl.DataFrame(
        {
            "sequence": ["MAKV"] * 10_001,
            "fitness": [0.0] * 10_001,
            "split": ["train"] * 10_001,
        }
    )

    with pytest.raises(ValidationError, match="at most 10,000"):
        Esm2GP().train(
            TrainContext(
                frame=frame,
                targets={"fitness": TaskType.REGRESSION},
                structure_column="sequence",
                conditions={},
                seed=7,
            )
        )


@needs_weights
@pytest.mark.parametrize("likelihood", [False, True])
def test_regression_trains_predicts_and_round_trips(likelihood: bool) -> None:
    import pickle

    from daikonstudio.application.engines.context import PredictContext, TrainContext
    from daikonstudio.application.engines.manifest import TaskType

    frame = pl.DataFrame(
        {
            "sequence": _SEQUENCES,
            "fitness": [0.5, 2.0, 4.0, 0.6, 1.0, 8.0, 0.4, 0.7],
            "split": ["train"] * 4 + ["validation", "validation", "test", "test"],
        }
    )
    engine = Esm2GP()
    result = engine.train(
        TrainContext(
            frame=frame,
            targets={"fitness": TaskType.REGRESSION},
            structure_column="sequence",
            conditions={"n_restarts_optimizer": 0, "likelihood_feature": likelihood},
            seed=7,
        )
    )

    assert "fitness" in result.metrics
    assert result.validation_metrics is not None
    bundle = pickle.loads(result.artifact)
    assert bundle["featurizer"] == ("esm2-650m+llr" if likelihood else "esm2-650m")
    assert bundle["parent"] == _SEQUENCES[0] if likelihood else "parent" not in bundle

    predictions = engine.predict(
        PredictContext(
            frame=pl.DataFrame({"sequence": _SEQUENCES[:3]}),
            structure_column="sequence",
            artifact=result.artifact,
            conditions={},
            target_columns=("fitness",),
        )
    )

    assert predictions.height == 3
    assert predictions["value"].null_count() == 0
    # The reason this engine exists: a posterior standard deviation, in the target's
    # units, for every row -- not a null the way the XGBoost path reports.
    assert predictions["uncertainty"].null_count() == 0
    assert (predictions["uncertainty"] > 0).all()


@needs_weights
def test_the_gp_varies_with_its_input_rather_than_returning_one_number() -> None:
    """The ALDE failure, asserted against, and asserted on the right quantity.

    A GP whose kernel has underflowed has no neighbours to borrow from, so it returns its
    prior mean for *every* row: the predictions become a constant. That is what to check.
    Not r2 -- on a test partition this small the target variance is a rounding error and
    r2 is enormous either way, which says nothing about whether the kernel works.
    """
    from daikonstudio.application.engines.context import PredictContext, TrainContext
    from daikonstudio.application.engines.manifest import TaskType

    targets = [0.5, 2.0, 4.0, 0.6, 1.0, 8.0, 0.4, 0.7]
    frame = pl.DataFrame(
        {"sequence": _SEQUENCES, "fitness": targets, "split": ["train"] * 6 + ["test"] * 2}
    )
    engine = Esm2GP()

    result = engine.train(
        TrainContext(
            frame=frame,
            targets={"fitness": TaskType.REGRESSION},
            structure_column="sequence",
            conditions={"n_restarts_optimizer": 0},
            seed=7,
        )
    )
    values = engine.predict(
        PredictContext(
            frame=pl.DataFrame({"sequence": _SEQUENCES}),
            structure_column="sequence",
            artifact=result.artifact,
            conditions={},
            target_columns=("fitness",),
        )
    )["value"].to_numpy()

    assert np.isfinite(values).all()
    # The targets span 0.4 to 8.0. A prior-only GP would put every row on one number.
    assert values.std() > 0.1, values
    # And the fitted length scale has to stay on the data's own scale rather than
    # collapsing to the bound, which is the same failure arriving by a different route.
    length_scale = pickle.loads(result.artifact)["model"]._gp.kernel_.k1.k2.length_scale
    assert 1.0 < length_scale < 1e4, length_scale
