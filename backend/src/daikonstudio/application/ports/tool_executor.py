"""Port for running one tool of the catalogue.

Inputs and outputs travel as blob references, not bytes: a tool step may run on a
runner, which reads and writes blobs through the API under the run's workspace.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolInput:
    key: str
    kind: str
    blob_key: str


@dataclass(frozen=True)
class ToolOutput:
    key: str
    kind: str
    blob_key: str
    summary: dict[str, Any]


@dataclass(frozen=True)
class ToolRunContext:
    workspace_id: uuid.UUID
    requested_by: uuid.UUID
    study_id: uuid.UUID
    step_id: uuid.UUID
    # Every output blob key starts with this; it starts with f"{workspace_id}/".
    output_prefix: str


class ToolExecutor(Protocol):
    async def available(self, tool_id: str) -> bool:
        """Async because a lane tool is available only while a runner serves its lane."""
        ...

    async def run(
        self,
        *,
        tool_id: str,
        params: Mapping[str, Any],
        inputs: Sequence[ToolInput],
        context: ToolRunContext,
    ) -> list[ToolOutput]:
        """`inputs` carries one entry per supplied slot; a missing optional one is absent."""
        ...
