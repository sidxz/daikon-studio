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


def test_conditions_reject_non_numeric_value_by_key_name():
    with pytest.raises(ValueError, match="n_estimators"):
        validate_conditions(MANIFEST, {"n_estimators": "lots"})


def test_registry_returns_registered_engine():
    registry = EngineRegistry({"fake": FakeEngine()})
    assert registry.get("fake").manifest().id == "fake"
    assert [m.id for m in registry.manifests()] == ["fake"]


def test_registry_raises_for_unknown_engine():
    with pytest.raises(UnknownEngineError):
        EngineRegistry({}).get("nope")
