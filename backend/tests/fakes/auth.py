import uuid
from dataclasses import dataclass, field


@dataclass
class FakeAuth:
    """Plain attributes satisfy the read-only-property Protocol; that direction is fine."""

    user_id: uuid.UUID = field(default_factory=uuid.uuid4)
    workspace_id: uuid.UUID = field(default_factory=uuid.uuid4)
    workspace_role: str = "editor"
