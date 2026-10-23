import io
import json
import uuid

import polars as pl

from daikonstudio.application.ports.tool_executor import ToolInput, ToolOutput
from daikonstudio.application.tools.codecs import encode
from daikonstudio.application.tools.tool_run import (
    OUTPUTS_MANIFEST,
    ToolRunParams,
    execute_tool_run,
    read_outputs,
    tool_run_prefix,
)
from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.infrastructure.tools.runtime import run_tool
from tests.fakes.blob_store import InMemoryBlobStore


def _params(ws: uuid.UUID, sid: uuid.UUID, run_id: uuid.UUID) -> ToolRunParams:
    return ToolRunParams(
        tool_id="standardize",
        lane="default",
        study_id=sid,
        step_id=uuid.uuid4(),
        params={},
        inputs=(
            ToolInput(
                key="compounds",
                kind="compound_table",
                blob_key=f"{ws}/studies/{sid}/values/x.parquet",
            ),
        ),
        output_prefix=tool_run_prefix(ws, sid, run_id),
    )


def test_params_round_trip_through_json() -> None:
    ws, sid, rid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    params = _params(ws, sid, rid)
    wire = json.loads(json.dumps(params.to_params()))
    assert wire["study_id"] == str(sid)
    assert ToolRunParams.from_params(wire) == params


def test_prefix_starts_with_the_workspace() -> None:
    ws, sid, rid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    assert tool_run_prefix(ws, sid, rid) == f"{ws}/studies/{sid}/tool-runs/{rid}/"


def test_execute_tool_run_writes_outputs_then_manifest() -> None:
    ws, sid = uuid.uuid4(), uuid.uuid4()
    store = InMemoryBlobStore()
    frame = pl.DataFrame({"SMILES": ["CCO", "c1ccccc1", "CCN"]})
    blob, _ = encode("compound_table", frame)
    store.put_bytes(f"{ws}/studies/{sid}/values/x.parquet", blob)
    rid = uuid.uuid4()
    params = _params(ws, sid, rid)
    run = Run(
        id=rid,
        kind=RunKind.TOOL,
        workspace_id=ws,
        requested_by=uuid.uuid4(),
        cache_key="k",
        params=params.to_params(),
    )
    prefix = params.output_prefix

    uri = execute_tool_run(run, store, run_tool)

    assert uri == f"memory://{prefix}{OUTPUTS_MANIFEST}"
    assert pl.read_parquet(io.BytesIO(store.blobs[f"{prefix}compounds.parquet"])).height == 3
    [entry] = json.loads(store.blobs[f"{prefix}{OUTPUTS_MANIFEST}"])
    assert entry["kind"] == "compound_table"
    assert entry["blob_key"].startswith(prefix)
    assert entry["summary"]["rows"] == 3
    assert read_outputs(store, prefix) == [ToolOutput(**entry)]
