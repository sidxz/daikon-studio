"""Duar auth integration — SDK initialization and FastAPI wiring."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from duar_auth import Duar

from daikonstudio.settings import Settings

if TYPE_CHECKING:
    from duar_auth import RequestAuth

    from daikonstudio.application.auth import AuthContext

logger = logging.getLogger(__name__)


if TYPE_CHECKING:

    def _static_check_request_auth_satisfies_auth_context(auth: RequestAuth) -> AuthContext:
        """mypy-only proof that RequestAuth still structurally matches AuthContext.

        Never executed — only exists for `mypy src` (run by `make lint`) to
        type-check the `return auth` below. If a future duar-auth bump
        renames/removes user_id/workspace_id/workspace_role, or turns one of them
        from a read-only @property into something incompatible, this assignment
        stops type-checking and `make lint` fails here — not as an AttributeError
        deep inside a route dependency in a later task.
        """
        return auth


# Service actions registered with Duar on startup — RBAC permissions that can
# be granted to workspace roles. No `studio:register_engine`: third-party engines
# land in Phase 5 and get their own admin-gated action plus a review step then.
SERVICE_ACTIONS = [
    "studio:read",
    "studio:write",
    "studio:train",
    "studio:publish",
    "studio:admin_config",
]


def build_duar(settings: Settings) -> Duar:
    # ponytail: actions are NOT passed to the constructor. The SDK lifespan would
    # register them synchronously and treat failure as fatal, turning rarely-changing
    # housekeeping into a boot blocker. Registered best-effort in the app lifespan
    # instead (see register_service_actions); the JWKS fetch stays fatal, as auth
    # genuinely cannot work without the signing key.
    return Duar(
        base_url=settings.duar_url,
        # Falls back to the display name when unset -- see Settings.duar_service_name
        # for why these are separate fields.
        service_name=settings.duar_service_name or settings.service_name,
        service_key=settings.duar_service_key,
        mode="authz",
        idp_jwks_url=settings.idp_jwks_url,
        idp_audience=settings.idp_audience,
        idp_issuer=settings.idp_issuer,
        cache_ttl=120,
    )


@lru_cache(maxsize=1)
def get_duar() -> Duar:
    """Process-wide singleton — app.py and interface/dependencies/_core.py must
    share exactly one Duar instance, never build one each.

    The SDK's `fetch_whoami()` (run once, in whichever instance's lifespan
    actually executes) re-points *that instance's* already-created
    PermissionClient/RoleClient at the shared realm scope by mutating them in
    place — duar.py's own docstring: "The get_auth dependency factory
    captures these instances by reference, so mutating .service_name in place
    updates that path too." A second, independently-built Duar would
    register service actions under the realm slug while every
    RequestAuth.check_action()/can() from *that* instance's get_auth still posts
    service_name="daikon-studio": registered under one scope, checked under
    another. It also leaks that second instance's httpx clients, since the SDK's
    lifespan only closes the instance it was given.

    Raises ValueError (from Duar.__init__) if the service key or IdP
    audience is missing. Deliberately loud: create_app() calls this
    unconditionally, and a service that can't authenticate should fail at boot,
    not serve traffic unprotected.
    """
    return build_duar(Settings())


async def register_service_actions(duar: Duar) -> bool:
    """Register this service's RBAC actions, best-effort, at startup.

    Action definitions change rarely and are not needed to serve requests, so a
    transient Duar slowdown/outage must never block boot. On failure we log
    and continue; a later successful startup re-registers. Returns True iff
    registration succeeded.
    """
    try:
        await duar.roles.register_actions([{"action": action} for action in SERVICE_ACTIONS])
    except Exception:
        logger.exception("duar actions registration failed (%d actions)", len(SERVICE_ACTIONS))
        return False
    logger.info("duar actions registered (%d actions)", len(SERVICE_ACTIONS))
    return True
