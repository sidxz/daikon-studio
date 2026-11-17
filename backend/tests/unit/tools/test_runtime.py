import io
import uuid

import polars as pl
import pytest
from pydantic import ValidationError

from daikonstudio.application.ports.tool_executor import ToolInput, ToolRunContext
from daikonstudio.infrastructure.tools.runtime import ChildProcessToolExecutor, run_tool
from tests.fakes.blob_store import InMemoryBlobStore
from tests.unit.tools.test_chem_tools import FRAME


def _parquet() -> bytes:
    buf = io.BytesIO()
    FRAME.write_parquet(buf)
    return buf.getvalue()


def test_run_tool_standardize() -> None:
    (out,) = run_tool("standardize", {}, {"compounds": _parquet()})
    assert (out.key, out.kind) == ("compounds", "compound_table")
    assert out.summary["rows"] == 4


def test_run_tool_validates_params() -> None:
    with pytest.raises(ValidationError):
        run_tool("fetch_pdb", {}, {})


def test_missing_output_key_is_named() -> None:
    with pytest.raises(KeyError, match="compounds"):
        run_tool(
            "standardize", {}, {"compounds": _parquet()}, impls={"standardize": lambda p, i: {}}
        )


WS = uuid.uuid4()
CONTEXT = ToolRunContext(
    workspace_id=WS,
    requested_by=uuid.uuid4(),
    study_id=uuid.uuid4(),
    step_id=uuid.uuid4(),
    output_prefix=f"{WS}/studies/s/values/",
)


async def test_child_process_matches_in_process() -> None:
    store = InMemoryBlobStore()
    store.put_bytes("in.parquet", _parquet())
    (child,) = await ChildProcessToolExecutor(store).run(
        tool_id="standardize",
        params={},
        inputs=[ToolInput(key="compounds", kind="compound_table", blob_key="in.parquet")],
        context=CONTEXT,
    )
    (local,) = run_tool("standardize", {}, {"compounds": _parquet()})
    assert child.blob_key.startswith(CONTEXT.output_prefix) and child.blob_key.endswith(".parquet")
    assert (child.key, child.kind, child.summary) == (local.key, local.kind, local.summary)
    assert store.get_bytes(child.blob_key) == local.data
    assert pl.read_parquet(io.BytesIO(local.data)).height == 4
    # Each output written once, beside the input it read.
    assert len(store.blobs) == 2
