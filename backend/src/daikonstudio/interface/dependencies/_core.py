"""Sentinel auth dependency — the auth dependency every protected route will use.

Lazy init: don't crash at import time if Sentinel env vars aren't set. Uses a
reject-all stub when Sentinel is unavailable so auth is never bypassed.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends

from daikonstudio.domain.shared.errors import ServiceUnavailableError
from daikonstudio.infrastructure.sentinel.auth import build_sentinel
from daikonstudio.settings import Settings

__all__ = ["AuthDep", "get_auth"]


async def _sentinel_not_configured() -> None:
    """Reject every request when Sentinel is not configured. Never a bypass."""
    raise ServiceUnavailableError(
        "Sentinel auth is not configured",
        detail="Set STUDIO_SENTINEL_SERVICE_KEY to enable authentication.",
    )


# Sentinel is "configured" only when the service key is explicitly set — the
# pydantic-settings default ("") is a missing-config signal, not a usable key.
_settings = Settings()
_get_request_auth: Any = (
    build_sentinel(_settings).get_auth
    if _settings.sentinel_service_key
    else _sentinel_not_configured
)


async def get_auth(auth: Annotated[Any, Depends(_get_request_auth)]) -> Any:
    """Stable auth dependency wrapper — overridable via dependency_overrides in tests."""
    return auth


AuthDep = Annotated[Any, Depends(get_auth)]
