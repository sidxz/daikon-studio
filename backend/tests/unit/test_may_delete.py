"""Who may delete a dataset or a draft protocol."""

import uuid
from dataclasses import dataclass

import pytest

from daikonstudio.application.auth import may_delete, require_may_delete
from daikonstudio.domain.shared.errors import AuthorizationError

CREATOR = uuid.uuid4()


@dataclass(frozen=True)
class _Auth:
    user_id: uuid.UUID
    workspace_id: uuid.UUID
    workspace_role: str


def _auth(role: str, user_id: uuid.UUID | None = None) -> _Auth:
    return _Auth(user_id=user_id or uuid.uuid4(), workspace_id=uuid.uuid4(), workspace_role=role)


@pytest.mark.parametrize(
    ("auth", "allowed"),
    [
        (_auth("editor", CREATOR), True),
        (_auth("editor"), False),
        (_auth("viewer", CREATOR), False),
        (_auth("admin"), True),
        (_auth("owner"), True),
    ],
)
def test_admins_and_the_creator_may_delete(auth, allowed):
    assert may_delete(auth, CREATOR) is allowed


def test_an_item_with_no_recorded_creator_is_admin_only():
    assert may_delete(_auth("editor"), None) is False
    assert may_delete(_auth("admin"), None) is True


def test_refusal_names_who_may_delete():
    with pytest.raises(AuthorizationError, match="Only an admin or the person who created it"):
        require_may_delete(_auth("editor"), CREATOR)
