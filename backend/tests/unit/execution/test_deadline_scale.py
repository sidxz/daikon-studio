"""A training run's time limit is one lane budget per fit it holds: the model's, the
random-split comparison's on a scaffold split, and the baseline's -- each once per
target for an engine that trains a model per target, and once per ensemble member."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from daikonstudio.application.engines.manifest import (
    ENSEMBLE_SIZE,
    ConditionSpec,
    ConditionType,
    EngineManifest,
    TaskType,
)
from daikonstudio.application.execution.train_protocol import deadline_scale
from daikonstudio.domain.data.split import SplitStrategy

_RANDOM: Any = SimpleNamespace(
    targets=["a", "b", "c"], split=SimpleNamespace(strategy=SplitStrategy.RANDOM)
)
_SCAFFOLD: Any = SimpleNamespace(
    targets=["a"], split=SimpleNamespace(strategy=SplitStrategy.SCAFFOLD)
)


def _manifest(*, joint: bool, ensemble: bool) -> EngineManifest:
    return EngineManifest(
        id="engine",
        version="1.0.0",
        name="Engine",
        description="",
        tasks=(TaskType.REGRESSION,),
        conditions=(
            ConditionSpec(
                key=ENSEMBLE_SIZE,
                label="Ensemble size",
                type=ConditionType.INTEGER,
                default=1,
                minimum=1,
                maximum=10,
            ),
        )
        if ensemble
        else (),
        supports_multitask=joint,
    )


_FOREST = EngineManifest(
    id="forest", version="1.0.0", name="Forest", description="", tasks=(TaskType.REGRESSION,)
)


def test_an_ensemble_counts_each_member() -> None:
    joint = _manifest(joint=True, ensemble=True)
    assert deadline_scale(joint, _RANDOM, {ENSEMBLE_SIZE: 4}, _FOREST, {}) == 4 + 3
    assert deadline_scale(joint, _RANDOM, {}, _FOREST, {}) == 1 + 3  # the default is one model


def test_a_scaffold_split_counts_the_random_split_comparison() -> None:
    """The prod run that timed out: an ensemble of four, trained twice, plus the baseline."""
    joint = _manifest(joint=True, ensemble=True)
    assert deadline_scale(joint, _SCAFFOLD, {ENSEMBLE_SIZE: 4}, _FOREST, {}) == 2 * 4 + 1


def test_an_ensemble_per_target_multiplies_both() -> None:
    # A form may post a number as a string; the worker coerces it, so this does too.
    per_target = _manifest(joint=False, ensemble=True)
    assert deadline_scale(per_target, _RANDOM, {ENSEMBLE_SIZE: "2"}, _FOREST, {}) == 6 + 3


def test_a_size_the_worker_will_refuse_counts_one_model() -> None:
    joint = _manifest(joint=True, ensemble=True)
    assert deadline_scale(joint, _RANDOM, {ENSEMBLE_SIZE: 50}, _FOREST, {}) == 1 + 3


def test_an_engine_without_the_setting_ignores_a_stray_one() -> None:
    per_target = _manifest(joint=False, ensemble=False)
    assert deadline_scale(per_target, _RANDOM, {ENSEMBLE_SIZE: 5}, _FOREST, {}) == 3 + 3


def test_an_ensemble_baseline_counts_its_members() -> None:
    """A chemprop baseline of ten models fits ten times inside the run."""
    joint = _manifest(joint=True, ensemble=True)
    single = _manifest(joint=True, ensemble=False)
    assert deadline_scale(single, _RANDOM, {}, joint, {ENSEMBLE_SIZE: 10}) == 1 + 10


# --- The optional legs ----------------------------------------------------------------
#
# A faithful reproduction runs the published protocol and nothing else. Our baseline and
# our random-split comparison are additions, so a scientist can switch them off -- and
# the budget has to follow, or the lane deadline kills a run that was never going to
# take that long.

_SIMPLE = _manifest(joint=False, ensemble=False)


def test_a_run_without_the_random_comparison_budgets_one_leg():
    assert deadline_scale(_SIMPLE, _SCAFFOLD, {}, _SIMPLE, {}, optimism_gap=False) == 1 + 1


def test_a_run_without_a_baseline_budgets_no_baseline_fit():
    assert deadline_scale(_SIMPLE, _SCAFFOLD, {}, _SIMPLE, {}, run_baseline=False) == 2


def test_both_off_on_a_grouped_split_is_a_single_fit():
    assert (
        deadline_scale(_SIMPLE, _SCAFFOLD, {}, _SIMPLE, {}, run_baseline=False, optimism_gap=False)
        == 1
    )


def test_switching_the_comparison_off_cannot_shrink_a_random_split_further():
    """A random split already skips the leg. The toggle must not subtract it twice."""
    assert deadline_scale(_SIMPLE, _RANDOM, {}, _SIMPLE, {}, optimism_gap=False) == 3 + 3


def test_both_default_to_on():
    assert deadline_scale(_SIMPLE, _SCAFFOLD, {}, _SIMPLE, {}) == 2 + 1
