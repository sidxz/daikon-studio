"""Workspace guards. Structural typing so the app never imports Sentinel SDK types."""

from __future__ import annotations

import uuid
from typing import Protocol, runtime_checkable

from daikonstudio.domain.shared.errors import AuthorizationError, NotFoundError

_ROLE_RANK = {"viewer": 0, "editor": 1, "admin": 2, "owner": 3}


@runtime_checkable
class AuthContext(Protocol):
    """Auth context available to use cases. Satisfied by Sentinel's RequestAuth."""

    user_id: uuid.UUID
    workspace_id: uuid.UUID
    workspace_role: str
    actions: frozenset[str]


def require_authenticated(auth: AuthContext | None) -> None:
    if auth is None:
        raise AuthorizationError("Authentication required")


def _require_rank(auth: AuthContext | None, minimum: str) -> None:
    if auth is None:  # system/worker call
        return
    if _ROLE_RANK.get(auth.workspace_role, -1) < _ROLE_RANK[minimum]:
        raise AuthorizationError(f"Requires {minimum} role or higher")


def require_editor(auth: AuthContext | None) -> None:
    _require_rank(auth, "editor")


def require_admin(auth: AuthContext | None) -> None:
    _require_rank(auth, "admin")


def require_same_workspace(
    auth: AuthContext | None, workspace_id: uuid.UUID, *, entity_type: str
) -> None:
    if auth is None:
        return
    if auth.workspace_id != workspace_id:
        # NotFound, not Authorization: 403 would confirm the resource exists.
        raise NotFoundError(entity_type)
