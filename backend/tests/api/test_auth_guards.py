import uuid

import pytest
from tests.fakes.auth import FakeAuth

from daikonstudio.application.auth import (
    require_admin,
    require_authenticated,
    require_editor,
    require_same_workspace,
)
from daikonstudio.domain.shared.errors import AuthorizationError, NotFoundError


def test_editor_guard_rejects_viewer():
    with pytest.raises(AuthorizationError):
        require_editor(FakeAuth(workspace_role="viewer"))


def test_editor_guard_allows_editor_and_admin():
    require_editor(FakeAuth(workspace_role="editor"))
    require_editor(FakeAuth(workspace_role="admin"))


def test_admin_guard_rejects_editor():
    with pytest.raises(AuthorizationError):
        require_admin(FakeAuth(workspace_role="editor"))


def test_cross_workspace_access_raises_not_found_not_forbidden():
    """403 would leak the existence of another workspace's resource."""
    auth = FakeAuth(workspace_id=uuid.uuid4())
    with pytest.raises(NotFoundError):
        require_same_workspace(auth, uuid.uuid4(), entity_type="Dataset")


def test_system_calls_bypass_role_guards():
    """auth=None is the worker calling a use case; roles do not apply."""
    require_editor(None)
    require_admin(None)


def test_authenticated_guard_rejects_none():
    """Unlike the role guards, require_authenticated has no system-call bypass."""
    with pytest.raises(AuthorizationError):
        require_authenticated(None)
