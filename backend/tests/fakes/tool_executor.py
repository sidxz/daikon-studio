"""An in-process `ToolExecutor`: the real tools, no child process."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.tool_executor import ToolInput, ToolOutput, ToolRunContext
from daikonstudio.application.tools.tool_run import RawOutput
from daikonstudio.infrastructure.tools.runtime import (
    IMPLEMENTATIONS,
    ChildProcessToolExecutor,
    ToolImpl,
    run_tool,
)


class InProcessToolExecutor(ChildProcessToolExecutor):
    """The child executor's blob reads and writes, with the tool run in this process."""

    def __init__(self, store: BlobStore, impls: Mapping[str, ToolImpl] | None = None) -> None:
        super().__init__(store)
        self.impls = IMPLEMENTATIONS if impls is None else impls
        self.calls: list[tuple[str, dict[str, Any], list[str]]] = []

    async def available(self, tool_id: str) -> bool:
        return tool_id in self.impls

    async def run(
        self,
        *,
        tool_id: str,
        params: Mapping[str, Any],
        inputs: Sequence[ToolInput],
        context: ToolRunContext,
    ) -> list[ToolOutput]:
        self.calls.append((tool_id, dict(params), [i.key for i in inputs]))
        return await super().run(tool_id=tool_id, params=params, inputs=inputs, context=context)

    async def _execute(
        self, tool_id: str, params: Mapping[str, Any], inputs: Mapping[str, bytes]
    ) -> list[RawOutput]:
        return await asyncio.to_thread(run_tool, tool_id, params, inputs, impls=self.impls)
