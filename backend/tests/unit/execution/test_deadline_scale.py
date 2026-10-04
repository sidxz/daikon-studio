"""A training run's time limit grows with the fits it holds: one per target for an
engine that trains a model per target, one per model for an ensemble."""

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

_THREE_TARGETS: Any = SimpleNamespace(targets=["a", "b", "c"])


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


def test_an_ensemble_scales_the_limit_by_its_size() -> None:
    joint = _manifest(joint=True, ensemble=True)
    assert deadline_scale(joint, _THREE_TARGETS, {ENSEMBLE_SIZE: 4}) == 4
    assert deadline_scale(joint, _THREE_TARGETS, {}) == 1  # the default is one model


def test_an_ensemble_per_target_multiplies_both() -> None:
    # A form may post a number as a string; the worker coerces it, so this does too.
    per_target = _manifest(joint=False, ensemble=True)
    assert deadline_scale(per_target, _THREE_TARGETS, {ENSEMBLE_SIZE: "2"}) == 6


def test_a_size_the_worker_will_refuse_leaves_the_limit_unscaled() -> None:
    joint = _manifest(joint=True, ensemble=True)
    assert deadline_scale(joint, _THREE_TARGETS, {ENSEMBLE_SIZE: 50}) == 1


def test_an_engine_without_the_setting_ignores_a_stray_one() -> None:
    per_target = _manifest(joint=False, ensemble=False)
    assert deadline_scale(per_target, _THREE_TARGETS, {ENSEMBLE_SIZE: 5}) == 3
