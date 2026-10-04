from dataclasses import replace

import polars as pl
import pytest

from daikonstudio.application.engines.context import TrainContext
from daikonstudio.application.engines.manifest import (
    ConditionSpec,
    ConditionType,
    EngineManifest,
    TaskType,
    reset_inapplicable_conditions,
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
    with pytest.raises(ValueError, match="Trees"):
        validate_conditions(MANIFEST, {"n_estimators": 5000})


def test_conditions_reject_unknown_keys():
    with pytest.raises(ValueError, match="does not accept these settings"):
        validate_conditions(MANIFEST, {"learning_rate": 0.1})


def test_conditions_reject_value_that_cannot_coerce():
    with pytest.raises(ValueError, match="Trees"):
        validate_conditions(MANIFEST, {"n_estimators": "lots"})


def test_conditions_coerce_numeric_string():
    # Values arrive over JSON from a form; a numeric field can legitimately show up
    # as a string. "500" is accepted and coerced to an int, not stored as a str.
    assert validate_conditions(MANIFEST, {"n_estimators": "500"}) == {"n_estimators": 500}


def test_conditions_reject_bool_for_numeric_condition():
    with pytest.raises(ValueError, match="Trees"):
        validate_conditions(MANIFEST, {"n_estimators": True})


def test_conditions_reject_non_integral_float_for_integer_condition():
    with pytest.raises(ValueError, match="Trees"):
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
    with pytest.raises(UnknownEngineError, match="More than one baseline"):
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


def test_every_registered_manifest_round_trips_through_json():
    """The manifest is the Phase 5 HTTP envelope: engines move out of process
    by *serving* their manifest instead of being imported (manifest.py's
    docstring). That exit stays open only while every manifest field is plain
    data. This trips the moment someone adds a non-serializable field -- an
    infrastructure object, a callable, a custom type -- to EngineManifest or
    ConditionSpec. If it fails, fix the field, not this test.
    """
    import json
    from dataclasses import asdict

    from daikonstudio.infrastructure.engines.registry import default_registry

    manifests = default_registry().manifests()
    assert manifests, "registry unexpectedly empty"
    for manifest in manifests:
        payload = json.loads(json.dumps(asdict(manifest)))
        assert payload["id"] == manifest.id
        assert payload["tasks"] == [task.value for task in manifest.tasks]
        for spec, raw in zip(manifest.conditions, payload["conditions"], strict=True):
            assert raw["key"] == spec.key
            assert raw["type"] == spec.type.value


def test_the_registry_loads_with_no_gpu_extra_installed():
    """The API tier and the default-lane worker install without chemprop, transformers,
    torch or CUDA, and must still serve every manifest -- otherwise the engine picker
    differs per deployment and Task 6's two-image split does not work.

    A subprocess, not a monkeypatched `sys.modules`: what this guards against is an
    import hoisted to *module* scope in `chemprop_dmpnn.py` or `molformer_xl.py`, and
    by the time this test runs the registry is long since imported. Only a fresh
    interpreter can tell.
    """
    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent(
        """
        import sys

        BLOCKED = ("chemprop", "transformers")

        class _NoGpuExtra:
            def find_spec(self, name, path=None, target=None):
                root = name.split(".")[0]
                if root in BLOCKED:
                    raise ImportError("blocked")
                return None

        sys.meta_path.insert(0, _NoGpuExtra())

        from daikonstudio.infrastructure.engines.registry import default_registry

        ids = sorted(m.id for m in default_registry().manifests())
        assert "chemprop-dmpnn" in ids, ids
        assert "molformer-xl" in ids, ids
        assert "torch" not in sys.modules, "torch imported at module scope"
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_report_defaults_to_a_no_op() -> None:
    """An engine that never calls `report` must still train. Both ECFP4 engines are
    exactly that: a single `.fit()` offers no yield point, so they are honestly not
    interruptible, and the contract must not force them to pretend otherwise."""
    ctx = TrainContext(
        frame=pl.DataFrame({"smiles": ["CCO"], "y": [1.0], "split": ["train"]}),
        targets={"y": TaskType.REGRESSION},
        structure_column="smiles",
        conditions={},
        seed=7,
    )

    assert ctx.report(0.5, "anything") is None


def _ctx(targets):
    return TrainContext(
        frame=pl.DataFrame(), targets=targets, structure_column="smiles", conditions={}, seed=1
    )


def test_a_single_target_context_names_its_target_and_task():
    ctx = _ctx({"y": TaskType.REGRESSION})
    assert ctx.target_columns == ("y",)
    assert ctx.target_column == "y"
    assert ctx.task is TaskType.REGRESSION


def test_target_column_raises_on_a_context_with_several_targets():
    ctx = _ctx({"a": TaskType.REGRESSION, "b": TaskType.REGRESSION})
    assert ctx.task is TaskType.REGRESSION
    with pytest.raises(ValueError, match="2 targets"):
        _ = ctx.target_column


def test_task_raises_when_the_targets_mix_kinds():
    with pytest.raises(ValueError, match="different kinds"):
        _ = _ctx({"a": TaskType.REGRESSION, "b": TaskType.BINARY_CLASSIFICATION}).task


def test_a_manifest_does_not_learn_several_targets_jointly_unless_it_says_so():
    assert MANIFEST.supports_multitask is False


def test_a_setting_for_a_task_the_dataset_lacks_is_reset_to_its_default():
    manifest = replace(
        MANIFEST,
        conditions=(
            *MANIFEST.conditions,
            ConditionSpec(
                key="weighting",
                label="Weighting",
                type=ConditionType.ENUM,
                default="none",
                options=("none", "balanced"),
                tasks=(TaskType.BINARY_CLASSIFICATION,),
            ),
        ),
    )
    supplied = {"n_estimators": 7, "weighting": "balanced", "typo": 1}

    regression = reset_inapplicable_conditions(manifest, supplied, {TaskType.REGRESSION})
    # Unknown keys are left for `validate_conditions` to reject, and other settings stay.
    assert regression == {"n_estimators": 7, "weighting": "none", "typo": 1}
    for tasks in ({TaskType.BINARY_CLASSIFICATION}, set(TaskType)):
        assert reset_inapplicable_conditions(manifest, supplied, tasks) == supplied


def test_option_labels_must_line_up_with_options():
    with pytest.raises(ValueError, match="2 option labels for 3 options"):
        ConditionSpec(
            key="mode",
            label="Mode",
            type=ConditionType.ENUM,
            options=("a", "b", "c"),
            option_labels=("A", "B"),
        )
