import uuid
from dataclasses import dataclass, field


@dataclass
class FakeAuth:
    user_id: uuid.UUID = field(default_factory=uuid.uuid4)
    workspace_id: uuid.UUID = field(default_factory=uuid.uuid4)
    workspace_role: str = "editor"
    actions: frozenset[str] = frozenset({"studio:read", "studio:write", "studio:train"})
