"""The engines this deployment ships, keyed by their own manifest ids.

A module-level tuple rather than a plugin scan: two in-tree engines is not a
discovery problem. `EngineRegistry.baseline()` raises unless exactly one of
them is flagged, so "the baseline is mandatory" is checked against the shipped
set the first time anything trains, not asserted in a comment.

Engines hold no per-run state -- everything they need arrives on the
TrainContext -- so one instance each is shared across every run.
"""

from __future__ import annotations

from daikonstudio.application.engines.protocol import Engine
from daikonstudio.application.engines.registry import EngineRegistry
from daikonstudio.infrastructure.engines.ecfp4_randomforest import Ecfp4RandomForest
from daikonstudio.infrastructure.engines.ecfp4_xgboost import Ecfp4XGBoost

_ENGINES: tuple[Engine, ...] = (Ecfp4RandomForest(), Ecfp4XGBoost())


def default_registry() -> EngineRegistry:
    return EngineRegistry({engine.manifest().id: engine for engine in _ENGINES})
