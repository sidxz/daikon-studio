"""HTTP adapter for the ``ChemCellar`` port: ChemCellar's own public API.

Modeled on ChemCellar's adapter for prot-cellar
(``cellar/infrastructure/prot_cellar/target_source.py``): the caller's Duar headers
are forwarded untouched and the service key is never sent.

A run's compounds are its measured molecules (readouts) together with its plated ones
(plate map), so a run whose plates are set up but not yet read can still be predicted.
Plate-map wells carry the structure and name but not the registration number, so
molecules seen only on plates get that from ``GET /molecules?ids=``. ChemCellar
re-points readouts and batches when it merges molecules, so a merged-away id cannot
appear here.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import date
from typing import Any

import httpx

from daikonstudio.application.ports.chemcellar import (
    CellarCompound,
    CellarProtocol,
    CellarRun,
    CellarRunCompounds,
)
from daikonstudio.domain.shared.errors import (
    AuthorizationError,
    NotFoundError,
    ServiceUnavailableError,
)

_PAGE_SIZE = 200  # ChemCellar's maximum.
_IDS_PER_LOOKUP = 100  # About 3.7 KB of query string.
# Readout data is unpaginated: one row per well and readout, so a large plate run is a
# large response.
_TIMEOUT_SECONDS = 60.0


class HttpChemCellar:
    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        self._client = client
        self._base = base_url.rstrip("/")

    async def list_protocols(
        self, *, forwarded_headers: Mapping[str, str]
    ) -> list[CellarProtocol]:
        found: list[CellarProtocol] = []
        cursor: str | None = None
        while True:
            params: dict[str, str | int] = {"limit": _PAGE_SIZE}
            if cursor:
                params["cursor"] = cursor
            page = await self._get("/api/v1/protocols", forwarded_headers, params)
            found.extend(
                CellarProtocol(id=uuid.UUID(item["id"]), name=item["name"])
                for item in page["items"]
            )
            cursor = page.get("next_cursor")
            if not cursor:
                return sorted(found, key=lambda protocol: protocol.name.casefold())

    async def list_runs(
        self, protocol_id: uuid.UUID, *, forwarded_headers: Mapping[str, str]
    ) -> list[CellarRun]:
        items = await self._get(
            f"/api/v1/protocols/{protocol_id}/runs",
            forwarded_headers,
            missing="ChemCellar protocol",
        )
        return [_run(item) for item in items]

    async def run_compounds(
        self, run_id: uuid.UUID, *, forwarded_headers: Mapping[str, str]
    ) -> CellarRunCompounds:
        run = await self._get(
            f"/api/v1/runs/{run_id}", forwarded_headers, missing="ChemCellar run"
        )
        protocol = await self._get(
            f"/api/v1/protocols/{run['protocol_id']}",
            forwarded_headers,
            missing="ChemCellar protocol",
        )
        readouts = await self._get(
            "/api/v1/readout-data", forwarded_headers, {"run_id": str(run_id)}
        )
        plate_map = await self._get(
            f"/api/v1/runs/{run_id}/plate-map", forwarded_headers, missing=None
        )

        compounds: dict[str, CellarCompound] = {}
        for row in readouts:
            molecule_id = row.get("molecule_id")
            if molecule_id and molecule_id not in compounds:
                compounds[molecule_id] = CellarCompound(
                    molecule_id=uuid.UUID(molecule_id),
                    registration_number=row.get("registration_number"),
                    name=row.get("molecule_name"),
                    smiles=row.get("smiles"),
                )

        plated: dict[str, dict[str, Any]] = {}
        for plate in (plate_map or {}).get("plates", []):
            for well in plate.get("wells", []):
                molecule_id = well.get("molecule_id")
                if molecule_id and molecule_id not in compounds:
                    plated.setdefault(molecule_id, well)
        ids = sorted(plated)
        registration: dict[str, str | None] = {}
        for start in range(0, len(ids), _IDS_PER_LOOKUP):
            chunk = ids[start : start + _IDS_PER_LOOKUP]
            page = await self._get(
                "/api/v1/molecules", forwarded_headers, {"ids": ",".join(chunk)}
            )
            registration.update(
                (item["id"], item.get("registration_number")) for item in page["items"]
            )
        for molecule_id in ids:
            well = plated[molecule_id]
            compounds[molecule_id] = CellarCompound(
                molecule_id=uuid.UUID(molecule_id),
                registration_number=registration.get(molecule_id),
                name=well.get("molecule_name"),
                smiles=well.get("smiles"),
            )

        return CellarRunCompounds(
            run_id=run_id,
            protocol_id=uuid.UUID(run["protocol_id"]),
            protocol_name=protocol["name"],
            run_date=date.fromisoformat(run["run_date"]),
            compounds=tuple(compounds.values()),
        )

    async def _get(
        self,
        path: str,
        headers: Mapping[str, str],
        params: Mapping[str, str | int] | None = None,
        *,
        missing: str | None = "ChemCellar record",
    ) -> Any:
        """GET and decode JSON. A 404 raises NotFoundError naming `missing`, or answers
        None when `missing` is None (a resource whose absence is ordinary)."""
        if not self._base:
            raise ServiceUnavailableError(
                "ChemCellar is not configured.",
                detail="Set STUDIO_CHEMCELLAR_API_URL to ChemCellar's API address.",
            )
        try:
            response = await self._client.get(
                f"{self._base}{path}",
                headers=dict(headers),
                params=dict(params or {}),
                timeout=_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            raise ServiceUnavailableError(
                "ChemCellar could not be reached.", detail=str(exc)
            ) from exc
        if response.status_code in (401, 403):
            raise AuthorizationError(
                "ChemCellar denied access to this data.",
                detail=f"({response.status_code}) {_detail(response)}",
            )
        if response.status_code == 404:
            if missing is None:
                return None
            raise NotFoundError(missing, path)
        if not response.is_success:
            raise ServiceUnavailableError(
                f"ChemCellar returned an error ({response.status_code}).",
                detail=_detail(response),
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ServiceUnavailableError(
                "ChemCellar returned an unreadable response.", detail=response.text[:200]
            ) from exc


def _run(item: Mapping[str, Any]) -> CellarRun:
    return CellarRun(
        id=uuid.UUID(item["id"]),
        protocol_id=uuid.UUID(item["protocol_id"]),
        run_date=date.fromisoformat(item["run_date"]),
        status=item["status"],
        measured_count=int(item.get("molecule_count") or 0),
        plate_count=int(item.get("plate_count") or 0),
        plate_barcodes=tuple(item.get("plate_barcodes") or ()),
    )


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict):
        return str(body.get("detail") or body.get("message") or body)[:200]
    return str(body)[:200]
