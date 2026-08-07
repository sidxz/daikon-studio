"""The engines this deployment ships, keyed by their own manifest ids.

A module-level tuple rather than a plugin scan: a handful of in-tree engines is not a
discovery problem. `EngineRegistry.baseline()` raises unless exactly one of
them is flagged, so "the baseline is mandatory" is checked against the shipped
set the first time anything trains, not asserted in a comment.

Every engine is registered in every deployment, including ones with no GPU. That is
deliberate: the manifest is plain data and costs nothing to serve, so the engine picker
stays identical everywhere, and a deployment that cannot run one finds out through a Run
that sits PENDING with no worker on its lane -- which is visible and diagnosable --
rather than through an engine that mysteriously does not appear.

Engines hold no per-run state -- everything they need arrives on the
TrainContext -- so one instance each is shared across every run.
"""

from __future__ import annotations

from daikonstudio.application.engines.protocol import Engine
from daikonstudio.application.engines.registry import EngineRegistry
from daikonstudio.infrastructure.engines.chemprop_dmpnn import ChempropDMPNN
from daikonstudio.infrastructure.engines.descriptors_xgboost import DescriptorsXGBoost
from daikonstudio.infrastructure.engines.ecfp4_lightgbm import Ecfp4LightGBM
from daikonstudio.infrastructure.engines.ecfp4_randomforest import Ecfp4RandomForest
from daikonstudio.infrastructure.engines.ecfp4_xgboost import Ecfp4XGBoost
from daikonstudio.infrastructure.engines.molformer_xl import MolformerXL
from daikonstudio.infrastructure.engines.tanimoto_gp import TanimotoGP

_ENGINES: tuple[Engine, ...] = (
    Ecfp4RandomForest(),
    Ecfp4XGBoost(),
    Ecfp4LightGBM(),
    DescriptorsXGBoost(),
    TanimotoGP(),
    ChempropDMPNN(),
    MolformerXL(),
)


def default_registry() -> EngineRegistry:
    return EngineRegistry({engine.manifest().id: engine for engine in _ENGINES})
