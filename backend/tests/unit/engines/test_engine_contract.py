from dataclasses import replace

import polars as pl
import pytest

from daikonstudio.application.engines.manifest import (
    ConditionSpec,
    ConditionType,
    EngineManifest,
    TaskType,
    validate_conditions,
)
from daikonstudio.application.engines.registry import EngineRegistry, UnknownEngineError

MANIFEST = EngineManifest(
    id="fake",
    version="1.0.0",
    name="Fake",
    description="test double",
    tasks=(TaskType.REGRESSION,),
    conditions=(
        ConditionSpec(
            key="n_estimators",
            label="Trees",
            type=ConditionType.INTEGER,
            default=100,
            minimum=1,
            maximum=1000,
        ),
    ),
)


class FakeEngine:
    @staticmethod
    def manifest() -> EngineManifest:
        return MANIFEST

    def train(self, ctx):
        return None

    def predict(self, ctx):
        return pl.DataFrame({"row_id": [], "value": []})


def test_conditions_fill_in_defaults():
    assert validate_conditions(MANIFEST, {}) == {"n_estimators": 100}


def test_conditions_reject_out_of_range():
    with pytest.raises(ValueError, match="n_estimators"):
        validate_conditions(MANIFEST, {"n_estimators": 5000})


def test_conditions_reject_unknown_keys():
    with pytest.raises(ValueError, match="unknown"):
        validate_conditions(MANIFEST, {"learning_rate": 0.1})


def test_conditions_reject_value_that_cannot_coerce():
    with pytest.raises(ValueError, match="n_estimators"):
        validate_conditions(MANIFEST, {"n_estimators": "lots"})


def test_conditions_coerce_numeric_string():
    # Values arrive over JSON from a form; a numeric field can legitimately show up
    # as a string. "500" is accepted and coerced to an int, not stored as a str.
    assert validate_conditions(MANIFEST, {"n_estimators": "500"}) == {"n_estimators": 500}


def test_conditions_reject_bool_for_numeric_condition():
    with pytest.raises(ValueError, match="n_estimators"):
        validate_conditions(MANIFEST, {"n_estimators": True})


def test_conditions_reject_non_integral_float_for_integer_condition():
    with pytest.raises(ValueError, match="n_estimators"):
        validate_conditions(MANIFEST, {"n_estimators": 500.5})


def test_registry_returns_registered_engine():
    registry = EngineRegistry({"fake": FakeEngine()})
    assert registry.get("fake").manifest().id == "fake"
    assert [m.id for m in registry.manifests()] == ["fake"]


def test_registry_raises_for_unknown_engine():
    with pytest.raises(UnknownEngineError):
        EngineRegistry({}).get("nope")


def _baseline_engine(engine_id: str):
    manifest = replace(MANIFEST, id=engine_id, is_baseline=True)

    class _BaselineEngine:
        @staticmethod
        def manifest():
            return manifest

        def train(self, ctx):
            return None

        def predict(self, ctx):
            return pl.DataFrame({"row_id": [], "value": []})

    return _BaselineEngine()


def test_registry_raises_for_multiple_baselines():
    registry = EngineRegistry({"a": _baseline_engine("a"), "b": _baseline_engine("b")})
    with pytest.raises(UnknownEngineError, match="multiple baseline"):
        registry.baseline()


def test_unknown_engine_error_maps_to_a_real_http_status_not_a_bare_500():
    """I6 (whole-branch review, Important): `EngineRegistry` is handed to the
    interface layer directly (`engines.py`'s route depends on the registry
    itself, not a Result-returning use case), so it was one route away from
    an unguarded `.get()`/`.baseline()` call. Before the fix,
    `UnknownEngineError` subclassed bare `KeyError`, the one error type in
    the app outside `error_handlers.py`'s single status map: `isinstance`
    against every entry there was `False`, and `_error_to_status` fell
    through to its unmapped-error default of 500 -- exactly the bare crash a
    scientist would see instead of a domain error naming the missing engine.
    """
    from daikonstudio.domain.shared.errors import DomainError
    from daikonstudio.interface.error_handlers import _error_to_status

    with pytest.raises(DomainError) as get_error:
        EngineRegistry({}).get("nope")
    assert _error_to_status(get_error.value) == 404

    with pytest.raises(DomainError) as baseline_error:
        EngineRegistry({}).baseline()
    assert _error_to_status(baseline_error.value) == 404
