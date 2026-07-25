"""Sentinel auth integration — SDK initialization and FastAPI wiring."""

from __future__ import annotations

import logging

from sentinel_auth import Sentinel

from daikonstudio.settings import Settings

logger = logging.getLogger(__name__)

# Service actions registered with Sentinel on startup — RBAC permissions that can
# be granted to workspace roles. No `studio:register_engine`: third-party engines
# land in Phase 5 and get their own admin-gated action plus a review step then.
SERVICE_ACTIONS = [
    "studio:read",
    "studio:write",
    "studio:train",
    "studio:publish",
    "studio:admin_config",
]


def build_sentinel(settings: Settings) -> Sentinel:
    # ponytail: actions are NOT passed to the constructor. The SDK lifespan would
    # register them synchronously and treat failure as fatal, turning rarely-changing
    # housekeeping into a boot blocker. Registered best-effort in the app lifespan
    # instead (see register_service_actions); the JWKS fetch stays fatal, as auth
    # genuinely cannot work without the signing key.
    return Sentinel(
        base_url=settings.sentinel_url,
        service_name=settings.service_name,
        service_key=settings.sentinel_service_key,
        mode="authz",
        idp_jwks_url=settings.idp_jwks_url,
        idp_audience=settings.idp_audience,
        idp_issuer=settings.idp_issuer,
        cache_ttl=120,
    )


async def register_service_actions(sentinel: Sentinel) -> bool:
    """Register this service's RBAC actions, best-effort, at startup.

    Action definitions change rarely and are not needed to serve requests, so a
    transient Sentinel slowdown/outage must never block boot. On failure we log
    and continue; a later successful startup re-registers. Returns True iff
    registration succeeded.
    """
    try:
        await sentinel.roles.register_actions([{"action": action} for action in SERVICE_ACTIONS])
    except Exception:
        logger.exception("sentinel actions registration failed (%d actions)", len(SERVICE_ACTIONS))
        return False
    logger.info("sentinel actions registered (%d actions)", len(SERVICE_ACTIONS))
    return True
