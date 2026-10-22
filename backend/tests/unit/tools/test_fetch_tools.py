from __future__ import annotations

import httpx
import pytest

from daikonstudio.infrastructure.tools.impl import fetch
from daikonstudio.infrastructure.tools.params import (
    FetchAlphafoldParams,
    FetchPdbParams,
    FetchUniprotParams,
)


def _atom(
    record: str,
    serial: int,
    name: str,
    res: str,
    chain: str,
    seq: int,
    xyz: tuple[float, float, float],
    b: float,
) -> str:
    return (
        f"{record:<6}{serial:>5} {name:<4} {res:>3} {chain}{seq:>4}    "
        f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}{1.0:>6.2f}{b:>6.2f}"
    )


PDB = "\n".join(
    [
        "HEADER    TEST",
        _atom("ATOM", 1, " N", "ALA", "A", 1, (0.0, 0.0, 0.0), 20.0),
        _atom("ATOM", 2, " CA", "ALA", "A", 1, (1.0, 0.0, 0.0), 20.0),
        _atom("HETATM", 3, " C1", "LIG", "A", 100, (4.0, 0.0, 0.0), 30.0),
        _atom("HETATM", 4, " O", "HOH", "A", 200, (1.0, 1.0, 0.0), 30.0),
        "END",
    ]
)
AF_PDB = "\n".join(
    [
        _atom("ATOM", 1, " N", "ALA", "A", 1, (0.0, 0.0, 0.0), 10.0),
        _atom("ATOM", 2, " CA", "ALA", "A", 1, (1.0, 0.0, 0.0), 90.0),
        _atom("ATOM", 3, " CA", "GLY", "A", 2, (4.0, 0.0, 0.0), 70.0),
        "END",
    ]
)
ENTRY = {
    "struct": {"title": "A test"},
    "exptl": [{"method": "X-RAY DIFFRACTION"}],
    "rcsb_entry_info": {"resolution_combined": [1.8]},
    "rcsb_entry_container_identifiers": {"polymer_entity_ids": ["1"]},
}
ENTITY = {"rcsb_polymer_entity_container_identifiers": {"uniprot_ids": ["P0AE18"]}}
UNIPROT = {
    "primaryAccession": "P0AE18",
    "proteinDescription": {"recommendedName": {"fullName": {"value": "MetAP"}}},
    "organism": {"scientificName": "E. coli"},
    "sequence": {"length": 264, "value": "MAISIKTPEDIEKMRVAGRLAAEVLEMIEP"},
    "features": [
        {
            "type": "Binding site",
            "location": {"start": {"value": 10}, "end": {"value": 12}},
            "description": "substrate",
        },
        {"type": "Chain", "location": {"start": {"value": 1}, "end": {"value": 264}}},
    ],
    "uniProtKBCrossReferences": [
        {"database": "PDB", "id": "4TRO"},
        {"database": "Pfam", "id": "X"},
    ],
}
SEARCH_URL = (
    "https://rest.uniprot.org/uniprotkb/search?query=methionine+aminopeptidase&size=1&format=json"
)
AF = [
    {"pdbUrl": "https://af.test/model.pdb", "latestVersion": 4, "modelCreatedDate": "2022-06-01"}
]


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen.append(url)
        routes: dict[str, httpx.Response] = {
            "https://files.rcsb.org/download/4TRO.pdb": httpx.Response(200, text=PDB),
            "https://data.rcsb.org/rest/v1/core/entry/4TRO": httpx.Response(200, json=ENTRY),
            "https://data.rcsb.org/rest/v1/core/polymer_entity/4TRO/1": httpx.Response(
                200, json=ENTITY
            ),
            "https://rest.uniprot.org/uniprotkb/P0AE18.json": httpx.Response(200, json=UNIPROT),
            SEARCH_URL: httpx.Response(200, json={"results": [{"primaryAccession": "P0AE18"}]}),
            "https://alphafold.ebi.ac.uk/api/prediction/P0AE18": httpx.Response(200, json=AF),
            "https://af.test/model.pdb": httpx.Response(200, text=AF_PDB),
        }
        return routes.get(url, httpx.Response(404))

    monkeypatch.setattr(fetch, "TRANSPORT", httpx.MockTransport(handler))
    return seen


def test_fetch_pdb_meta(calls: list[str]) -> None:
    out = fetch.fetch_pdb(FetchPdbParams(pdb_id="4tro"), {})
    s = out["structure"]
    assert s["format"] == "pdb" and s["text"] == PDB
    meta = s["meta"]
    assert meta["id"] == "4TRO" and meta["source"] == "rcsb"
    assert meta["resolution"] == 1.8 and meta["method"] == "X-RAY DIFFRACTION"
    assert meta["chains"] == ["A"] and meta["uniprot"] == ["P0AE18"]
    assert meta["predicted"] is False
    assert len(meta["ligands"]) == 1
    lig = meta["ligands"][0]
    assert lig["comp_id"] == "LIG" and lig["contacts"] == [
        {"chain": "A", "resname": "ALA", "resseq": 1}
    ]


def test_fetch_pdb_missing(calls: list[str]) -> None:
    with pytest.raises(ValueError, match="4XXX"):
        fetch.fetch_pdb(FetchPdbParams(pdb_id="4xxx"), {})


def test_fetch_uniprot_from_structure(calls: list[str]) -> None:
    structure = fetch.fetch_pdb(FetchPdbParams(pdb_id="4TRO"), {})["structure"]
    out = fetch.fetch_uniprot(FetchUniprotParams(), {"structure": structure})["annotations"]
    assert out["meta"]["accession"] == "P0AE18" and out["meta"]["length"] == 264
    assert out["meta"]["name"] == "MetAP" and out["meta"]["organism"] == "E. coli"
    assert out["features"] == [
        {"type": "Binding site", "start": 10, "end": 12, "description": "substrate"}
    ]
    assert out["structures"] == ["4TRO"]
    assert out["meta"]["features"] == 1 and out["meta"]["structures"] == 1


def test_fetch_uniprot_no_accession() -> None:
    with pytest.raises(ValueError, match="No accession given"):
        fetch.fetch_uniprot(FetchUniprotParams(), {})


def test_fetch_uniprot_name_searches_first(calls: list[str]) -> None:
    fetch.fetch_uniprot(FetchUniprotParams(query="methionine aminopeptidase"), {})
    assert "uniprotkb/search" in calls[0]


def test_fetch_alphafold(calls: list[str]) -> None:
    meta = fetch.fetch_alphafold(FetchAlphafoldParams(accession="P0AE18"), {})["structure"]["meta"]
    assert meta["predicted"] is True and meta["mean_plddt"] == 80.0
    assert (
        meta["source"] == "alphafold" and meta["ligands"] == [] and meta["uniprot"] == ["P0AE18"]
    )
