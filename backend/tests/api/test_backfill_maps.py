from tests.api import test_triage_round_trip

from daikonstudio.infrastructure.backfill_maps import backfill
from daikonstudio.infrastructure.chem.chemical_space import UmapLayout
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer
from daikonstudio.infrastructure.storage.fsspec_blob_store import FsspecBlobStore

# Shared with the triage suite, bound here so pytest finds it.
ready_run_id = test_triage_round_trip.ready_run_id


async def test_backfill_rebuilds_what_is_missing_and_then_has_nothing_to_do(
    client, tmp_path, workspace_id, session_factory, ready_run_id
):
    run = (await client.get(f"/api/v1/runs/{ready_run_id}")).json()
    protocol_id = run["protocol_id"]
    base = tmp_path / str(workspace_id)
    (base / "protocols" / protocol_id / "chemical-space.json").unlink()
    (base / "runs" / ready_run_id / "neighbors.parquet").unlink()
    store = FsspecBlobStore(f"file://{tmp_path}")

    def quiet(_: str) -> None:
        return None

    first = await backfill(
        session_factory, store, UmapLayout(), RdkitStructureNormalizer(), log=quiet
    )
    assert (first.maps_built, first.neighbours_built, first.failures) == (1, 1, [])
    assert (base / "protocols" / protocol_id / "chemical-space.json").exists()
    assert (base / "runs" / ready_run_id / "neighbors.parquet").exists()

    second = await backfill(
        session_factory, store, UmapLayout(), RdkitStructureNormalizer(), log=quiet
    )
    assert (second.maps_built, second.neighbours_built) == (0, 0)
    assert (second.maps_current, second.neighbours_current) == (1, 1)
