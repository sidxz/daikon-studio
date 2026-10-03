"""Workspace guards. Structural typing so the app never imports Duar SDK types."""

from __future__ import annotations

import uuid
from typing import Protocol, runtime_checkable

from daikonstudio.domain.shared.errors import AuthorizationError, NotFoundError

_ROLE_RANK = {"viewer": 0, "editor": 1, "admin": 2, "owner": 3}


@runtime_checkable
class AuthContext(Protocol):
    """Structural match for the SDK's RequestAuth without importing it.

    Members are declared as read-only properties, not plain variables: RequestAuth
    implements them as @property, and mypy treats a plain variable in a Protocol as
    read-write, so a read-only property would not satisfy it.

    There is deliberately no `actions` member. RequestAuth has no such attribute —
    action checks go through its async `check_action()`, a different shape entirely.
    No guard here needs it.
    """

    @property
    def user_id(self) -> uuid.UUID: ...

    @property
    def workspace_id(self) -> uuid.UUID: ...

    @property
    def workspace_role(self) -> str: ...


def require_authenticated(auth: AuthContext | None) -> None:
    if auth is None:
        raise AuthorizationError("Authentication required")


def _require_rank(auth: AuthContext | None, minimum: str) -> None:
    if auth is None:  # system/worker call
        return
    if _ROLE_RANK.get(auth.workspace_role, -1) < _ROLE_RANK[minimum]:
        raise AuthorizationError(f"This action requires the {minimum} role or higher.")


def require_editor(auth: AuthContext | None) -> None:
    _require_rank(auth, "editor")


def require_admin(auth: AuthContext | None) -> None:
    _require_rank(auth, "admin")


def is_editor(auth: AuthContext | None) -> bool:
    """For the UI: whether this viewer may change a dataset's settings."""
    return auth is None or _ROLE_RANK.get(auth.workspace_role, -1) >= _ROLE_RANK["editor"]


def require_same_workspace(
    auth: AuthContext | None, workspace_id: uuid.UUID, *, entity_type: str
) -> None:
    if auth is None:
        return
    if auth.workspace_id != workspace_id:
        # NotFound, not Authorization: 403 would confirm the resource exists.
        raise NotFoundError(entity_type)


def may_delete(auth: AuthContext | None, created_by: uuid.UUID | None) -> bool:
    """Admins and owners may delete anything; the creator may delete their own
    while they still hold editor or higher. `created_by` is None for items made
    before creators were recorded (migration 011), so those are admin-only."""
    if auth is None:  # system/worker call
        return True
    rank = _ROLE_RANK.get(auth.workspace_role, -1)
    if rank >= _ROLE_RANK["admin"]:
        return True
    return created_by is not None and created_by == auth.user_id and rank >= _ROLE_RANK["editor"]


def require_may_delete(auth: AuthContext | None, created_by: uuid.UUID | None) -> None:
    if not may_delete(auth, created_by):
        raise AuthorizationError("Only an admin or the person who created it can delete this.")
