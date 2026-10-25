"""Runs one tool of the catalogue: validate params, decode inputs, run, encode outputs.

`ChildProcessToolExecutor` does that in a spawned child, for the reason
`infrastructure/chem/profile.py` gives: RDKit work holds the API's GIL.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from pydantic import BaseModel

from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.tool_executor import ToolInput, ToolOutput, ToolRunContext
from daikonstudio.application.tools.codecs import decode, encode, extension
from daikonstudio.application.tools.tool_run import RawOutput
from daikonstudio.infrastructure.tools.catalog import default_registry
from daikonstudio.infrastructure.tools.impl import (
    chem,
    cores,
    fetch,
    library,
    pairs,
    structure,
)

ToolImpl = Callable[[BaseModel | None, Mapping[str, object]], Mapping[str, object]]

IMPLEMENTATIONS: dict[str, ToolImpl] = {
    **chem.IMPLEMENTATIONS,
    **library.IMPLEMENTATIONS,
    **fetch.IMPLEMENTATIONS,
    **structure.IMPLEMENTATIONS,
    **cores.IMPLEMENTATIONS,
    **pairs.IMPLEMENTATIONS,
}


def run_tool(
    tool_id: str,
    params: Mapping[str, Any],
    inputs: Mapping[str, bytes],
    *,
    impls: Mapping[str, ToolImpl] = IMPLEMENTATIONS,
) -> list[RawOutput]:
    manifest = default_registry().get(tool_id)
    validated = manifest.params.model_validate(params) if manifest.params else None
    decoded = {
        slot.key: decode(slot.kind.value, inputs[slot.key])
        for slot in manifest.inputs
        if slot.key in inputs
    }
    produced = impls[tool_id](validated, decoded)
    # A tool may return `meta` beside its outputs: what it found, said once, for a reader
    # that gets a summary and fifty rows. A table has nowhere to carry it -- its blob is a
    # frame -- so it is folded into the first output's summary here. Without this the
    # counts, the named rows and the refusals a table tool computed are thrown away, and
    # the step reports a shape with no finding in it.
    extra = produced.get("meta")
    outputs: list[RawOutput] = []
    for slot in manifest.outputs:
        if slot.key not in produced:
            raise KeyError(f"Tool {tool_id} did not produce its output {slot.key!r}")
        blob, summary = encode(slot.kind.value, produced[slot.key])
        if not outputs and isinstance(extra, Mapping) and "meta" not in summary:
            summary = {**summary, "meta": dict(extra)}
        outputs.append(RawOutput(key=slot.key, kind=slot.kind.value, data=blob, summary=summary))
    return outputs


class ChildProcessToolExecutor:
    """Reads the inputs here, runs the tool in a child, writes each output once under
    `context.output_prefix`."""

    def __init__(self, store: BlobStore) -> None:
        self._store = store

    async def available(self, tool_id: str) -> bool:
        return tool_id in IMPLEMENTATIONS

    async def run(
        self,
        *,
        tool_id: str,
        params: Mapping[str, Any],
        inputs: Sequence[ToolInput],
        context: ToolRunContext,
    ) -> list[ToolOutput]:
        data = {i.key: await asyncio.to_thread(self._store.get_bytes, i.blob_key) for i in inputs}
        outputs = []
        for raw in await self._execute(tool_id, params, data):
            blob_key = f"{context.output_prefix}{uuid.uuid4()}.{extension(raw.kind)}"
            await asyncio.to_thread(self._store.put_bytes, blob_key, raw.data)
            outputs.append(
                ToolOutput(key=raw.key, kind=raw.kind, blob_key=blob_key, summary=raw.summary)
            )
        return outputs

    async def _execute(
        self, tool_id: str, params: Mapping[str, Any], inputs: Mapping[str, bytes]
    ) -> list[RawOutput]:
        spawn = multiprocessing.get_context("spawn")
        # A pool per call, as `ChildProcessProfileComputer` does: the child exits with
        # its memory instead of idling in a pool.
        pool = ProcessPoolExecutor(max_workers=1, mp_context=spawn)
        try:
            outputs = await asyncio.wrap_future(
                pool.submit(_run_in_child, tool_id, dict(params), dict(inputs))
            )
        except BaseException:
            # Cancelled or failed: kill the worker, because `shutdown(wait=False)` leaves
            # a running item running and the interpreter's exit would join it
            # (see `ChildProcessProfileComputer.compute`).
            for process in (pool._processes or {}).values():
                if process is not None:
                    process.terminate()
            pool.shutdown(wait=False)
            raise
        pool.shutdown(wait=True)
        return outputs


def _run_in_child(
    tool_id: str, params: dict[str, Any], inputs: dict[str, bytes]
) -> list[RawOutput]:
    """Module-level because `spawn` re-imports this module to find it."""
    return run_tool(tool_id, params, inputs)
