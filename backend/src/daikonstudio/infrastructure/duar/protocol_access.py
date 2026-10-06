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


def _unavailable(error: Exception, operation: str) -> ServiceUnavailableError:
    # The 503 the caller sees is deliberately generic; the log keeps Duar's own answer.
    _logger.warning(
        "protocol_acl_read_failed",
        operation=operation,
        status=getattr(error, "status_code", None),
        error=str(error),
    )
    return ServiceUnavailableError(_UNAVAILABLE)


def _request_auth(auth: AuthContext) -> RequestAuth:
    # At runtime every user-facing `auth` is the SDK's RequestAuth (AuthDep).
    return cast("RequestAuth", auth)


class DuarProtocolAccess:
    def __init__(self, duar: Duar, *, retry_delays: tuple[float, ...] = (0.5, 2.0)) -> None:
        self._duar = duar
        self._retry_delays = retry_delays

    async def register(self, protocol: InSilicoProtocol) -> None:
        # Never raises: a lost ACL row is repairable (`register_protocols`), a lost run is not.
        if protocol.created_by is None:
            _logger.warning("protocol_acl_no_creator", protocol_id=str(protocol.id))
            return
        visibility = "workspace" if protocol.is_locked else "private"
        try:
            await self._register(protocol, visibility, protocol.created_by)
        except (DuarError, httpx.HTTPError) as error:
            if _transient(error):
                # ponytail: reconciled at the next API boot (`register_on_boot`);
                # add a periodic pass if hidden-until-restart ever hurts.
                _logger.error(
                    "protocol_acl_register_failed", protocol_id=str(protocol.id), error=str(error)
                )
            else:  # e.g. 400 "Owner is not a member": admins still see it.
                _logger.warning(
                    "protocol_acl_register_refused", protocol_id=str(protocol.id), error=str(error)
                )
        except Exception as error:
            _logger.error(
                "protocol_acl_register_failed", protocol_id=str(protocol.id), error=str(error)
            )

    async def _register(
        self, protocol: InSilicoProtocol, visibility: str, owner_id: uuid.UUID
    ) -> None:
        """Register with retries on transient failures. Raises whatever finally fails."""
        for delay in (*self._retry_delays, None):
            try:
                await self._duar.permissions.register_resource(
                    resource_type=RESOURCE_TYPE,
                    resource_id=protocol.id,
                    workspace_id=protocol.workspace_id,
                    owner_id=owner_id,
                    visibility=visibility,
                )
                return
            except (DuarError, httpx.HTTPError) as error:
                if delay is None or not _transient(error):
                    raise
                await asyncio.sleep(delay)

    async def deregister(self, protocol_id: uuid.UUID) -> None:
        try:
            await self._duar.permissions.deregister_resource(RESOURCE_TYPE, protocol_id)
        except DuarError as error:
            if error.status_code == 404:
                return
            _logger.warning(
                "protocol_acl_deregister_failed", protocol_id=str(protocol_id), error=str(error)
            )
        except Exception as error:  # never raises, as `register`
            _logger.warning(
                "protocol_acl_deregister_failed", protocol_id=str(protocol_id), error=str(error)
            )

    async def make_workspace_visible(self, auth: AuthContext, protocol: InSilicoProtocol) -> None:
        try:
            await _request_auth(auth).update_visibility(RESOURCE_TYPE, protocol.id, "workspace")
        except DuarError as error:
            if error.status_code == 404:  # never registered: register it published
                await self._register_published(auth, protocol)
                return
            raise _unavailable(error, "make_workspace_visible") from error
        except httpx.HTTPError as error:
            raise _unavailable(error, "make_workspace_visible") from error

    async def _register_published(self, auth: AuthContext, protocol: InSilicoProtocol) -> None:
        # Raises on failure, so the publish fails and the protocol stays a draft. The
        # publisher is a current member, so they own it when its creator is gone.
        owner = protocol.created_by or auth.user_id
        try:
            try:
                await self._register(protocol, "workspace", owner)
            except DuarError as error:
                if _transient(error) or owner == auth.user_id:
                    raise
                _logger.warning(
                    "protocol_acl_register_refused", protocol_id=str(protocol.id), error=str(error)
                )
                await self._register(protocol, "workspace", auth.user_id)
        except Exception as error:
            raise _unavailable(error, "register_published") from error

    async def visible_ids(self, auth: AuthContext) -> frozenset[uuid.UUID] | None:
        try:
            ids, full = await _request_auth(auth).accessible(RESOURCE_TYPE, "view")
        except (DuarError, httpx.HTTPError) as error:
            raise _unavailable(error, "accessible") from error
        return None if full else frozenset(ids)

    async def can_view(self, auth: AuthContext, protocol_id: uuid.UUID) -> bool:
        try:
            return await _request_auth(auth).can(RESOURCE_TYPE, protocol_id, "view")
        except (DuarError, httpx.HTTPError) as error:
            raise _unavailable(error, "can") from error
