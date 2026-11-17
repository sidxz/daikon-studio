"""The three fetch tools: PDB, UniProt and AlphaFold.

Each takes the validated params and the decoded inputs and returns plain dicts.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

import httpx
import numpy as np
from pydantic import BaseModel

from daikonstudio.application.tools.structures import ca_chains, ligands, parse_pdb
from daikonstudio.infrastructure.tools.params import (
    FetchAlphafoldParams,
    FetchPdbParams,
    FetchUniprotParams,
)

ToolImpl = Callable[[BaseModel | None, Mapping[str, object]], Mapping[str, object]]

TRANSPORT: httpx.BaseTransport | None = None
RCSB_FILE = "https://files.rcsb.org/download/{id}.pdb"
RCSB_ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{id}"
RCSB_ENTITY = "https://data.rcsb.org/rest/v1/core/polymer_entity/{id}/{entity}"
UNIPROT_ENTRY = "https://rest.uniprot.org/uniprotkb/{accession}.json"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
ALPHAFOLD = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"

_ACCESSION = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
)
_FEATURES = {
    "Binding site",
    "Active site",
    "Site",
    "Metal binding",
    "Natural variant",
    "Mutagenesis",
}


def _get(
    client: httpx.Client, url: str, missing: str | None = None, **kwargs: Any
) -> httpx.Response:
    response = client.get(url, **kwargs)
    if missing and response.status_code == 404:
        raise ValueError(missing)
    if not response.is_success:
        raise ValueError(f"Request failed with status {response.status_code}: {url}")
    return response


def _client() -> httpx.Client:
    return httpx.Client(transport=TRANSPORT, timeout=60, follow_redirects=True)


def fetch_pdb(params: BaseModel | None, inputs: Mapping[str, object]) -> Mapping[str, object]:
    assert isinstance(params, FetchPdbParams)
    pdb_id = params.pdb_id
    missing = f"PDB entry {pdb_id} was not found."
    with _client() as client:
        # ponytail: PDB format only; entries lacking a PDB-format file fail with a clear message.
        text = _get(client, RCSB_FILE.format(id=pdb_id), missing).text
        entry = _get(client, RCSB_ENTRY.format(id=pdb_id), missing).json()
        uniprot: list[str] = []
        for entity in entry.get("rcsb_entry_container_identifiers", {}).get(
            "polymer_entity_ids", []
        ):
            data = _get(client, RCSB_ENTITY.format(id=pdb_id, entity=entity)).json()
            for acc in data.get("rcsb_polymer_entity_container_identifiers", {}).get(
                "uniprot_ids", []
            ):
                if acc not in uniprot:
                    uniprot.append(acc)
    atoms = parse_pdb(text)
    resolution = (entry.get("rcsb_entry_info", {}).get("resolution_combined") or [None])[0]
    meta = {
        "id": pdb_id,
        "source": "rcsb",
        "title": entry.get("struct", {}).get("title"),
        "method": (entry.get("exptl") or [{}])[0].get("method"),
        "resolution": resolution,
        "chains": ca_chains(atoms),
        "ligands": ligands(atoms),
        "uniprot": uniprot,
        "predicted": False,
        "mean_plddt": None,
    }
    return {"structure": {"format": "pdb", "text": text, "meta": meta}}


def fetch_uniprot(params: BaseModel | None, inputs: Mapping[str, object]) -> Mapping[str, object]:
    assert isinstance(params, FetchUniprotParams)
    with _client() as client:
        query = params.query
        if query is None:
            structure = inputs.get("structure")
            # An attached structure names no accession, so its meta may not have the key.
            accessions = (
                structure["meta"].get("uniprot", []) if isinstance(structure, dict) else []
            )
            if not accessions:
                raise ValueError("No accession given and no structure to read one from.")
            accession = accessions[0]
        elif _ACCESSION.match(query):
            accession = query
        else:
            hits = _get(
                client, UNIPROT_SEARCH, params={"query": query, "size": 1, "format": "json"}
            ).json()["results"]
            if not hits:
                raise ValueError(f"UniProt has no entry matching {query!r}.")
            accession = hits[0]["primaryAccession"]
        entry = _get(client, UNIPROT_ENTRY.format(accession=accession)).json()
    features = [
        {
            "type": f["type"],
            "start": f["location"]["start"]["value"],
            "end": f["location"]["end"]["value"],
            "description": f.get("description", ""),
        }
        for f in entry.get("features", [])
        if f["type"] in _FEATURES
    ]
    structures = [
        x["id"] for x in entry.get("uniProtKBCrossReferences", []) if x["database"] == "PDB"
    ]
    meta = {
        "accession": accession,
        "name": entry.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value"),
        "organism": entry.get("organism", {}).get("scientificName"),
        "length": entry.get("sequence", {}).get("length"),
        "features": len(features),
        "structures": len(structures),
    }
    return {
        "annotations": {
            "meta": meta,
            "sequence": entry.get("sequence", {}).get("value", ""),
            "features": features,
            "structures": structures,
        }
    }


def fetch_alphafold(
    params: BaseModel | None, inputs: Mapping[str, object]
) -> Mapping[str, object]:
    assert isinstance(params, FetchAlphafoldParams)
    accession = params.accession
    with _client() as client:
        listing = _get(
            client,
            ALPHAFOLD.format(accession=accession),
            f"AlphaFold has no model for {accession}.",
        )
        first = listing.json()[0]
        text = _get(client, first["pdbUrl"]).text
    atoms = parse_pdb(text)
    plddt = [a["b"] for a in atoms if a["name"] == "CA"]
    meta = {
        "id": accession,
        "source": "alphafold",
        "title": f"AlphaFold model of {accession}",
        "method": "AlphaFold",
        "resolution": None,
        "chains": ca_chains(atoms),
        "ligands": [],
        "uniprot": [accession],
        "predicted": True,
        "mean_plddt": float(np.mean(plddt)) if plddt else None,
        "version": first.get("latestVersion"),
        "created": first.get("modelCreatedDate"),
    }
    return {"structure": {"format": "pdb", "text": text, "meta": meta}}


IMPLEMENTATIONS: dict[str, ToolImpl] = {
    "fetch_pdb": fetch_pdb,
    "fetch_uniprot": fetch_uniprot,
    "fetch_alphafold": fetch_alphafold,
}
