from __future__ import annotations

from daikonstudio.application.engines.manifest import EngineManifest
from daikonstudio.application.engines.protocol import Engine


class UnknownEngineError(KeyError):
    pass


class EngineRegistry:
    def __init__(self, engines: dict[str, Engine]) -> None:
        self._engines = engines

    def get(self, engine_id: str) -> Engine:
        try:
            return self._engines[engine_id]
        except KeyError as exc:
            raise UnknownEngineError(engine_id) from exc

    def manifests(self) -> list[EngineManifest]:
        return [engine.manifest() for engine in self._engines.values()]

    def baseline(self) -> Engine:
        for engine in self._engines.values():
            if engine.manifest().is_baseline:
                return engine
        raise UnknownEngineError("no baseline engine registered")
