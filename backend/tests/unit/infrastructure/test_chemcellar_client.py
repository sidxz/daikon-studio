import uuid

import httpx
import pytest

from daikonstudio.domain.shared.errors import (
    AuthorizationError,
    NotFoundError,
    ServiceUnavailableError,
)
from daikonstudio.infrastructure.chemcellar.client import HttpChemCellar

HEADERS = {"authorization": "Bearer idp", "x-authz-token": "authz"}
RUN = str(uuid.uuid4())
PROTOCOL = str(uuid.uuid4())
M1, M2 = str(uuid.uuid4()), str(uuid.uuid4())


def _run_json(**overrides):
    return {
        "id": RUN,
        "protocol_id": PROTOCOL,
        "run_date": "2026-06-05",
        "status": "draft",
        "molecule_count": 1,
        "plate_count": 1,
        "plate_barcodes": ["P1"],
        **overrides,
    }


def _cellar(routes, seen=None, base="http://cellar"):
    """`routes` maps a path to a JSON body or an int status; `seen` collects requests."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        answer = routes.get(request.url.path)
        if answer is None:
            return httpx.Response(404, json={"detail": "not found"})
        if isinstance(answer, int):
            return httpx.Response(answer, json={"detail": "nope"})
        return httpx.Response(200, json=answer(request) if callable(answer) else answer)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HttpChemCellar(client, base)


def _compound_routes(plate_wells, molecules=lambda request: {"items": [], "next_cursor": None}):
    return {
        f"/api/v1/runs/{RUN}": _run_json(),
        f"/api/v1/protocols/{PROTOCOL}": {"id": PROTOCOL, "name": "NadD-Sumo dose response"},
        "/api/v1/readout-data": [
            {
                "molecule_id": M1,
                "registration_number": "CV-1",
                "molecule_name": "one",
                "smiles": "CCO",
            },
            {
                "molecule_id": M1,
                "registration_number": "CV-1",
                "molecule_name": "one",
                "smiles": "CCO",
            },
            {
                "molecule_id": None,
                "registration_number": None,
                "molecule_name": None,
                "smiles": None,
            },
        ],
        f"/api/v1/runs/{RUN}/plate-map": {"plates": [{"wells": plate_wells}]},
        "/api/v1/molecules": molecules,
    }


async def test_a_runs_compounds_are_its_measured_and_plated_molecules_once_each():
    seen: list[httpx.Request] = []
    wells = [
        {"molecule_id": M1, "molecule_name": "one", "smiles": "CCO"},
        {"molecule_id": M2, "molecule_name": "two", "smiles": "CCN"},
        {"molecule_id": None, "molecule_name": None, "smiles": None},
    ]
    molecules = lambda request: {  # noqa: E731
        "items": [{"id": M2, "registration_number": "CV-2"}],
        "next_cursor": None,
    }
    found = await _cellar(_compound_routes(wells, molecules), seen).run_compounds(
        uuid.UUID(RUN), forwarded_headers=HEADERS
    )

    assert found.protocol_name == "NadD-Sumo dose response"
    assert str(found.run_date) == "2026-06-05"
    assert sorted((c.registration_number, c.name, c.smiles) for c in found.compounds) == [
        ("CV-1", "one", "CCO"),
        ("CV-2", "two", "CCN"),
    ]
    lookup = [r for r in seen if r.url.path == "/api/v1/molecules"]
    assert [r.url.params["ids"] for r in lookup] == [M2]  # only the plated-only molecule
    for request in seen:  # the user's own headers, and never a service key
        assert request.headers["authorization"] == "Bearer idp"
        assert request.headers["x-authz-token"] == "authz"
        assert "x-service-key" not in request.headers


async def test_plated_only_registration_lookups_go_in_chunks_of_100():
    plated = [str(uuid.uuid4()) for _ in range(250)]
    wells = [{"molecule_id": m, "molecule_name": None, "smiles": "C"} for m in plated]
    seen: list[httpx.Request] = []
    await _cellar(_compound_routes(wells), seen).run_compounds(
        uuid.UUID(RUN), forwarded_headers=HEADERS
    )
    sizes = [
        len(r.url.params["ids"].split(",")) for r in seen if r.url.path == "/api/v1/molecules"
    ]
    assert sizes == [100, 100, 50]


async def test_a_run_without_a_plate_map_still_has_its_measured_compounds():
    routes = _compound_routes([])
    routes[f"/api/v1/runs/{RUN}/plate-map"] = 404
    found = await _cellar(routes).run_compounds(uuid.UUID(RUN), forwarded_headers=HEADERS)
    assert [c.registration_number for c in found.compounds] == ["CV-1"]


async def test_protocols_follow_every_page_and_sort_by_name():
    def protocols(request):
        if request.url.params.get("cursor") == "next":
            return {"items": [{"id": str(uuid.uuid4()), "name": "alpha"}], "next_cursor": None}
        return {"items": [{"id": str(uuid.uuid4()), "name": "Beta"}], "next_cursor": "next"}

    found = await _cellar({"/api/v1/protocols": protocols}).list_protocols(
        forwarded_headers=HEADERS
    )
    assert [p.name for p in found] == ["alpha", "Beta"]


async def test_a_protocols_runs_carry_counts():
    routes = {f"/api/v1/protocols/{PROTOCOL}/runs": [_run_json(molecule_count=60)]}
    [run] = await _cellar(routes).list_runs(uuid.UUID(PROTOCOL), forwarded_headers=HEADERS)
    assert (run.measured_count, run.plate_count, run.plate_barcodes) == (60, 1, ("P1",))


@pytest.mark.parametrize("status", [401, 403])
async def test_a_refusal_is_an_authorization_error(status):
    with pytest.raises(AuthorizationError) as raised:
        await _cellar({"/api/v1/protocols": status}).list_protocols(forwarded_headers=HEADERS)
    assert f"ChemCellar answered {status}: nope" in str(raised.value.detail)


async def test_a_missing_run_is_not_found():
    with pytest.raises(NotFoundError):
        await _cellar({}).run_compounds(uuid.UUID(RUN), forwarded_headers=HEADERS)


async def test_a_server_error_is_unavailable():
    with pytest.raises(ServiceUnavailableError):
        await _cellar({"/api/v1/protocols": 500}).list_protocols(forwarded_headers=HEADERS)


async def test_an_unreachable_chemcellar_is_unavailable():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    cellar = HttpChemCellar(httpx.AsyncClient(transport=httpx.MockTransport(refuse)), "http://x")
    with pytest.raises(ServiceUnavailableError) as raised:
        await cellar.list_protocols(forwarded_headers=HEADERS)
    assert not raised.value.detail


async def test_an_unconfigured_chemcellar_is_unavailable_without_a_request():
    seen: list[httpx.Request] = []
    with pytest.raises(ServiceUnavailableError, match="not configured"):
        await _cellar({}, seen, base="").list_protocols(forwarded_headers=HEADERS)
    assert seen == []
