"""Sentinel auth dependency — the auth dependency every protected route will use.

Lazy init: don't crash at import time if Sentinel env vars aren't set. Uses a
reject-all stub when Sentinel is unavailable so auth is never bypassed.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends
from pydantic import ValidationError

from daikonstudio.application.auth import AuthContext
from daikonstudio.domain.shared.errors import ServiceUnavailableError
from daikonstudio.infrastructure.sentinel.auth import get_sentinel

__all__ = ["AuthDep", "get_auth"]


async def _sentinel_not_configured() -> AuthContext:
    """Reject every request when Sentinel is not configured. Never a bypass."""
    raise ServiceUnavailableError(
        "Sentinel auth is not configured",
        detail="Set STUDIO_SENTINEL_SERVICE_KEY to enable authentication.",
    )


# get_sentinel() is the one process-wide Sentinel instance shared with
# interface/app.py, and raises ValueError when required settings (service key,
# IdP audience) are missing — deliberately loud there, since create_app() must
# fail at boot rather than serve traffic unprotected. Here we're defensive
# instead: this module can be imported independently of create_app() (e.g. by a
# route module under test), so a missing/malformed config falls back to the
# reject-all stub rather than propagating the crash — but never to a bypass.
try:
    _get_request_auth: Any = get_sentinel().get_auth
except (ValueError, ValidationError):
    _get_request_auth = _sentinel_not_configured


async def get_auth(auth: Annotated[AuthContext, Depends(_get_request_auth)]) -> AuthContext:
    """Stable auth dependency wrapper — overridable via dependency_overrides in tests."""
    return auth


AuthDep = Annotated[AuthContext, Depends(get_auth)]
