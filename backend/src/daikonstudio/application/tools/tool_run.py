"""A tool step as a queued Run: its params on the row, its outputs under a prefix.

`execute_tool_run` is what a runner does with a TOOL run: read each input blob, run
the tool, write each output, then write `outputs.json` listing them. The manifest goes
last, so a reader that finds it finds every output it names.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.tool_executor import ToolInput, ToolOutput
from daikonstudio.application.tools.codecs import extension
from daikonstudio.domain.execution.run import Run

OUTPUTS_MANIFEST = "outputs.json"


@dataclass(frozen=True)
class RawOutput:
    key: str
    kind: str
    data: bytes
    summary: dict[str, Any]


RunTool = Callable[[str, Mapping[str, Any], Mapping[str, bytes]], list[RawOutput]]


@dataclass(frozen=True)
class ToolRunParams:
    tool_id: str
    lane: str
    study_id: uuid.UUID
    position: int
    params: dict[str, Any]
    inputs: tuple[ToolInput, ...]
    output_prefix: str

    def to_params(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "lane": self.lane,
            "study_id": str(self.study_id),
            "position": self.position,
            "params": dict(self.params),
            "inputs": [asdict(i) for i in self.inputs],
            "output_prefix": self.output_prefix,
        }

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> ToolRunParams:
        return cls(
            tool_id=params["tool_id"],
            lane=params["lane"],
            study_id=uuid.UUID(str(params["study_id"])),
            position=int(params["position"]),
            params=dict(params["params"]),
            inputs=tuple(ToolInput(**i) for i in params["inputs"]),
            output_prefix=params["output_prefix"],
        )


def tool_run_prefix(workspace_id: uuid.UUID, study_id: uuid.UUID, run_id: uuid.UUID) -> str:
    return f"{workspace_id}/studies/{study_id}/tool-runs/{run_id}/"


def execute_tool_run(run: Run, store: BlobStore, run_tool: RunTool) -> str:
    """Returns the manifest's URI, as `put_bytes` gave it."""
    spec = ToolRunParams.from_params(run.params)
    # The API's runner blob endpoint enforces this too; checked here so an inline
    # execution cannot write where a runner could not.
    if not spec.output_prefix.startswith(f"{run.workspace_id}/"):
        raise ValueError("A tool run's outputs must be written under its own workspace.")
    inputs = {i.key: store.get_bytes(i.blob_key) for i in spec.inputs}
    outputs = []
    for raw in run_tool(spec.tool_id, spec.params, inputs):
        blob_key = f"{spec.output_prefix}{raw.key}.{extension(raw.kind)}"
        store.put_bytes(blob_key, raw.data)
        outputs.append(
            ToolOutput(key=raw.key, kind=raw.kind, blob_key=blob_key, summary=raw.summary)
        )
    manifest = json.dumps([asdict(o) for o in outputs], default=str).encode()
    return store.put_bytes(f"{spec.output_prefix}{OUTPUTS_MANIFEST}", manifest)


def read_outputs(store: BlobStore, prefix: str) -> list[ToolOutput]:
    return [ToolOutput(**o) for o in json.loads(store.get_bytes(f"{prefix}{OUTPUTS_MANIFEST}"))]
