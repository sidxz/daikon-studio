"""`ProtocolAccess` backed by Duar resource permissions. See the port's docstring."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, cast

import httpx
import structlog
from duar_auth import Duar, DuarError

from daikonstudio.application.auth import AuthContext
from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from daikonstudio.domain.shared.errors import ServiceUnavailableError

if TYPE_CHECKING:
    from duar_auth import RequestAuth

RESOURCE_TYPE = "studio_protocol"  # prefixed: realm apps share one namespace in Duar
_logger = structlog.get_logger(__name__)
_UNAVAILABLE = "Protocol permissions could not be checked. Try again in a moment."


def _transient(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPError):
        return True
    return isinstance(error, DuarError) and (error.status_code is None or error.status_code >= 500)


def _request_auth(auth: AuthContext) -> RequestAuth:
    # At runtime every user-facing `auth` is the SDK's RequestAuth (AuthDep).
    return cast("RequestAuth", auth)


class DuarProtocolAccess:
    def __init__(self, duar: Duar, *, retry_delays: tuple[float, ...] = (0.5, 2.0)) -> None:
        self._duar = duar
        self._retry_delays = retry_delays

    async def register(self, protocol: InSilicoProtocol) -> None:
        await self._register(protocol, "workspace" if protocol.is_locked else "private")

    async def _register(self, protocol: InSilicoProtocol, visibility: str) -> None:
        if protocol.created_by is None:
            _logger.warning("protocol_acl_no_creator", protocol_id=str(protocol.id))
            return
        for delay in (*self._retry_delays, None):
            try:
                await self._duar.permissions.register_resource(
                    resource_type=RESOURCE_TYPE,
                    resource_id=protocol.id,
                    workspace_id=protocol.workspace_id,
                    owner_id=protocol.created_by,
                    visibility=visibility,
                )
                return
            except (DuarError, httpx.HTTPError) as error:
                if not _transient(error):
                    # e.g. 400 "Owner is not a member": admins still see it.
                    _logger.warning(
                        "protocol_acl_register_refused",
                        protocol_id=str(protocol.id),
                        error=str(error),
                    )
                    return
                if delay is None:
                    # ponytail: no automatic reconcile; re-run `register_protocols` by hand.
                    _logger.error(
                        "protocol_acl_register_failed",
                        protocol_id=str(protocol.id),
                        error=str(error),
                    )
                    return
                await asyncio.sleep(delay)

    async def deregister(self, protocol_id: uuid.UUID) -> None:
        try:
            await self._duar.permissions.deregister_resource(RESOURCE_TYPE, protocol_id)
        except (DuarError, httpx.HTTPError) as error:
            if isinstance(error, DuarError) and error.status_code == 404:
                return
            _logger.warning(
                "protocol_acl_deregister_failed", protocol_id=str(protocol_id), error=str(error)
            )

    async def make_workspace_visible(self, auth: AuthContext, protocol: InSilicoProtocol) -> None:
        try:
            await _request_auth(auth).update_visibility(RESOURCE_TYPE, protocol.id, "workspace")
        except DuarError as error:
            if error.status_code == 404:  # never registered: register it published
                await self._register(protocol, "workspace")
                return
            raise ServiceUnavailableError(_UNAVAILABLE) from error
        except httpx.HTTPError as error:
            raise ServiceUnavailableError(_UNAVAILABLE) from error

    async def visible_ids(self, auth: AuthContext) -> frozenset[uuid.UUID] | None:
        try:
            ids, full = await _request_auth(auth).accessible(RESOURCE_TYPE, "view")
        except (DuarError, httpx.HTTPError) as error:
            raise ServiceUnavailableError(_UNAVAILABLE) from error
        return None if full else frozenset(ids)

    async def can_view(self, auth: AuthContext, protocol_id: uuid.UUID) -> bool:
        try:
            return await _request_auth(auth).can(RESOURCE_TYPE, protocol_id, "view")
        except (DuarError, httpx.HTTPError) as error:
            raise ServiceUnavailableError(_UNAVAILABLE) from error
