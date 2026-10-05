import uuid
from datetime import date

import httpx

from daikonstudio.application.ports.chemcellar import (
    CellarCompound,
    CellarProtocol,
    CellarRun,
    CellarRunCompounds,
    ChemCellar,
)
from daikonstudio.infrastructure.chemcellar.client import HttpChemCellar

PROTOCOL, RUN = uuid.uuid4(), uuid.uuid4()


class FakeChemCellar:
    def __init__(self):
        self.headers: list[dict[str, str]] = []

    async def list_protocols(self, *, forwarded_headers):
        self.headers.append(dict(forwarded_headers))
        return [CellarProtocol(id=PROTOCOL, name="NadD-Sumo dose response")]

    async def list_runs(self, protocol_id, *, forwarded_headers):
        return [
            CellarRun(
                id=RUN,
                protocol_id=protocol_id,
                run_date=date(2026, 6, 5),
                status="draft",
                measured_count=60,
                plate_count=1,
                plate_barcodes=("P1",),
            )
        ]

    async def run_compounds(self, run_id, *, forwarded_headers):
        return CellarRunCompounds(
            run_id=run_id,
            protocol_id=PROTOCOL,
            protocol_name="NadD-Sumo dose response",
            run_date=date(2026, 6, 5),
            compounds=(
                CellarCompound(
                    molecule_id=uuid.uuid4(), registration_number="CV-1", name=None, smiles="CCO"
                ),
            ),
        )


def _use(app, fake):
    app.state.container.define(ChemCellar, fake)


async def test_protocols_are_read_with_the_callers_own_headers(app, client):
    fake = FakeChemCellar()
    _use(app, fake)
    response = await client.get("/api/v1/chemcellar/protocols")
    assert response.status_code == 200
    assert response.json() == [{"id": str(PROTOCOL), "name": "NadD-Sumo dose response"}]
    assert set(fake.headers[0]) == {"authorization", "x-authz-token"}


async def test_a_protocols_runs(app, client):
    _use(app, FakeChemCellar())
    response = await client.get(f"/api/v1/chemcellar/protocols/{PROTOCOL}/runs")
    assert response.json()[0]["measured_count"] == 60
    assert response.json()[0]["run_date"] == "2026-06-05"


async def test_an_import_answers_with_an_upload_ref_and_its_source(app, client):
    _use(app, FakeChemCellar())
    response = await client.post(f"/api/v1/chemcellar/runs/{RUN}/import")
    assert response.status_code == 201
    body = response.json()
    assert body["compound_count"] == 1 and body["sample"] == ["CCO"]
    assert body["source"]["protocol_name"] == "NadD-Sumo dose response"
    uuid.UUID(body["upload_ref"])


async def test_a_viewer_cannot_import(app, viewer_client):
    _use(app, FakeChemCellar())
    response = await viewer_client.post(f"/api/v1/chemcellar/runs/{RUN}/import")
    assert response.status_code == 403


async def test_an_unconfigured_chemcellar_answers_503(app, client):
    # Explicit, not the default Settings: a developer's backend/.env may set the URL.
    _use(app, HttpChemCellar(httpx.AsyncClient(), ""))
    response = await client.get("/api/v1/chemcellar/protocols")
    assert response.status_code == 503
