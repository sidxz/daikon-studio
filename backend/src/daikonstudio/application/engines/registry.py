from __future__ import annotations

from daikonstudio.application.engines.manifest import EngineManifest
from daikonstudio.application.engines.protocol import Engine
from daikonstudio.domain.shared.errors import DomainError, NotFoundError


class UnknownEngineError(NotFoundError):
    """Raised by `.get()` for an unrecognized engine id, and by `.baseline()`
    when the registry isn't wired with exactly one baseline engine.

    Subclasses `NotFoundError`, not bare `KeyError` (whole-branch review,
    Important 6): `EngineRegistry` is handed to the interface layer directly
    (`engines.py`'s route depends on the registry itself, not on a
    Result-returning use case), so this was the one error type in the app
    outside `interface/error_handlers.py`'s `_ERROR_STATUS_MAP` -- the first
    route or use case that let a `.get()`/`.baseline()` call raise unguarded
    would have surfaced a bare, unmapped 500 rather than a domain error a
    client can act on.

    Deliberately calls `DomainError.__init__` rather than
    `NotFoundError.__init__`: the two call sites below don't share
    `NotFoundError`'s `(entity_type, entity_id)` shape -- `.get()`'s message
    is a bare engine id, `.baseline()`'s is a sentence describing the
    registry's own misconfiguration (no baseline registered, or more than
    one) -- and forcing the second through `"{entity_type} '{entity_id}' not
    found"` would misdescribe it. Inheriting from `NotFoundError` is only so
    `_error_to_status`'s `isinstance` check resolves this to 404 -- the
    least-wrong status available for either case (`.baseline()`'s two modes
    are a deployment misconfiguration, not `ServiceUnavailableError`'s "a
    reachable dependency is temporarily down"; nothing today reaches
    `.baseline()` from an HTTP-facing path unguarded, so the distinction
    between the two raise sites has no live status-code consequence yet).
    """

    def __init__(self, message: str) -> None:
        DomainError.__init__(self, message)


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
        baselines = [e for e in self._engines.values() if e.manifest().is_baseline]
        if not baselines:
            raise UnknownEngineError("no baseline engine registered")
        if len(baselines) > 1:
            ids = sorted(e.manifest().id for e in baselines)
            raise UnknownEngineError(f"multiple baseline engines registered: {ids}")
        return baselines[0]
