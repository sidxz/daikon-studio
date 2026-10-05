import json
import uuid
from datetime import date

import pytest

from daikonstudio.application.data.create_dataset import upload_key, upload_source_key
from daikonstudio.application.data.import_chemcellar_run import ImportChemCellarRun
from daikonstudio.application.ports.chemcellar import CellarCompound, CellarRunCompounds
from daikonstudio.domain.shared.errors import AuthorizationError, ValidationError
from tests.fakes.auth import FakeAuth
from tests.fakes.blob_store import InMemoryBlobStore

RUN, PROTOCOL = uuid.uuid4(), uuid.uuid4()


def _compound(reg, smiles, name=None):
    return CellarCompound(
        molecule_id=uuid.uuid4(), registration_number=reg, name=name, smiles=smiles
    )


class FakeChemCellar:
    def __init__(self, compounds):
        self.compounds = compounds
        self.headers = None

    async def run_compounds(self, run_id, *, forwarded_headers):
        self.headers = forwarded_headers
        return CellarRunCompounds(
            run_id=run_id,
            protocol_id=PROTOCOL,
            protocol_name="NadD-Sumo dose response",
            run_date=date(2026, 6, 5),
            compounds=tuple(self.compounds),
        )


async def _import(compounds, auth=None, store=None):
    store = store or InMemoryBlobStore()
    auth = auth or FakeAuth()
    cellar = FakeChemCellar(compounds)
    result = await ImportChemCellarRun(cellar, store)(
        RUN, forwarded_headers={"authorization": "Bearer t"}, auth=auth
    )
    return result.unwrap(), store, auth, cellar


async def test_an_import_is_an_ordinary_upload_with_its_source_beside_it():
    compounds = [
        _compound("CV-2", "CCN", "two"),
        _compound("CV-1", "CCO"),
        _compound("CV-3", None),
    ]
    imported, store, auth, cellar = await _import(compounds)

    csv_text = store.get_bytes(upload_key(auth.workspace_id, imported.upload_ref)).decode()
    assert csv_text == "smiles,compound_id,name\nCCO,CV-1,\nCCN,CV-2,two\n"
    assert (imported.compound_count, imported.without_structure) == (2, 1)
    assert imported.sample == ("CCO", "CCN")
    source = json.loads(store.get_bytes(upload_source_key(auth.workspace_id, imported.upload_ref)))
    assert source == {
        "app": "chemcellar",
        "run_id": str(RUN),
        "protocol_id": str(PROTOCOL),
        "protocol_name": "NadD-Sumo dose response",
        "run_date": "2026-06-05",
        "compounds_without_structure": 1,
    }
    assert imported.source == source
    assert cellar.headers == {"authorization": "Bearer t"}


async def test_the_same_run_imports_to_the_same_bytes_whatever_order_chemcellar_answers_in():
    compounds = [_compound(f"CV-{i}", "C" * i) for i in range(1, 6)]
    first, store_a, auth, _ = await _import(compounds)
    second, store_b, _, _ = await _import(list(reversed(compounds)), auth=auth)
    assert store_a.get_bytes(upload_key(auth.workspace_id, first.upload_ref)) == store_b.get_bytes(
        upload_key(auth.workspace_id, second.upload_ref)
    )


async def test_a_run_with_no_disclosed_structure_cannot_be_imported():
    store, auth = InMemoryBlobStore(), FakeAuth()
    result = await ImportChemCellarRun(FakeChemCellar([_compound("CV-1", None)]), store)(
        RUN, forwarded_headers={}, auth=auth
    )
    assert isinstance(result.failure(), ValidationError)


async def test_a_viewer_cannot_import():
    with pytest.raises(AuthorizationError):
        await ImportChemCellarRun(FakeChemCellar([]), InMemoryBlobStore())(
            RUN, forwarded_headers={}, auth=FakeAuth(workspace_role="viewer")
        )


async def test_a_run_with_no_compounds_says_so():
    result = await ImportChemCellarRun(FakeChemCellar([]), InMemoryBlobStore())(
        RUN, forwarded_headers={}, auth=FakeAuth()
    )
    failure = result.failure()
    assert isinstance(failure, ValidationError)
    assert "no compounds yet" in str(failure)
