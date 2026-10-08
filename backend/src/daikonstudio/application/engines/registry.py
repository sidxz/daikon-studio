from __future__ import annotations

from daikonstudio.application.engines.fan_out import FanOut
from daikonstudio.application.engines.manifest import EngineManifest
from daikonstudio.application.engines.protocol import Engine
from daikonstudio.domain.data.structure_kind import StructureKind
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
    names the missing engine id, `.baseline()`'s is a sentence describing the
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


def _uniform(engine: Engine) -> Engine:
    """Every engine as one that accepts several targets: as itself when it learns
    them jointly, inside `FanOut` otherwise. Applied at lookup so no caller can
    reach an unwrapped single-target engine by accident."""
    return engine if engine.manifest().supports_multitask else FanOut(engine)


class EngineRegistry:
    def __init__(self, engines: dict[str, Engine]) -> None:
        self._engines = engines

    def get(self, engine_id: str) -> Engine:
        try:
            return _uniform(self._engines[engine_id])
        except KeyError as exc:
            raise UnknownEngineError(
                f"Engine '{engine_id}' is not available on this server."
            ) from exc

    def manifests(self) -> list[EngineManifest]:
        return [engine.manifest() for engine in self._engines.values()]

    def baseline(self, structure_kind: StructureKind = StructureKind.MOLECULE) -> Engine:
        """The baseline for this kind of structure column.

        Keyed by kind rather than global, because a baseline has to be able to read the
        data it is the floor for. The molecule baseline on a sequence dataset is not a
        weak comparison, it is an impossible one -- `_check_capable` refuses it, and
        since the baseline is mandatory that refusal failed the entire run.
        """
        baselines = [
            e
            for e in self._engines.values()
            if e.manifest().is_baseline and structure_kind in e.manifest().structure_kinds
        ]
        if not baselines:
            raise UnknownEngineError(
                f"No baseline engine on this server can read {structure_kind.value} "
                f"data, and every Protocol is measured against a baseline."
            )
        if len(baselines) > 1:
            ids = sorted(e.manifest().id for e in baselines)
            raise UnknownEngineError(
                f"More than one baseline engine is configured for {structure_kind.value} "
                f"data: {', '.join(ids)}."
            )
        return _uniform(baselines[0])
