"""ChemCellar, the sibling app that holds assay runs, read as a compound source.

Every call forwards the requesting user's own Duar headers (``authorization`` and
``x-authz-token``). Both apps sit in the same Duar realm, so ChemCellar accepts
Studio's tokens as they are and applies its own workspace and role checks. Studio
holds no credential of its own for ChemCellar.

Failures are raised, not returned: ``AuthorizationError`` when ChemCellar refuses,
``NotFoundError`` for a missing run or protocol, ``ServiceUnavailableError`` when it
is unreachable, misbehaving or not configured.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Protocol


@dataclass(frozen=True, kw_only=True)
class CellarProtocol:
    id: uuid.UUID
    name: str


@dataclass(frozen=True, kw_only=True)
class CellarRun:
    id: uuid.UUID
    protocol_id: uuid.UUID
    run_date: date
    status: str
    # ChemCellar's own `molecule_count`: compounds with at least one readout.
    measured_count: int
    plate_count: int
    plate_barcodes: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class CellarCompound:
    molecule_id: uuid.UUID
    registration_number: str | None
    name: str | None
    # None when ChemCellar does not disclose the structure.
    smiles: str | None


@dataclass(frozen=True, kw_only=True)
class CellarRunCompounds:
    run_id: uuid.UUID
    protocol_id: uuid.UUID
    protocol_name: str
    run_date: date
    # One per molecule: plated or measured in the run.
    compounds: tuple[CellarCompound, ...]


class ChemCellar(Protocol):
    async def list_protocols(
        self, *, forwarded_headers: Mapping[str, str]
    ) -> list[CellarProtocol]: ...

    async def list_runs(
        self, protocol_id: uuid.UUID, *, forwarded_headers: Mapping[str, str]
    ) -> list[CellarRun]: ...

    async def run_compounds(
        self, run_id: uuid.UUID, *, forwarded_headers: Mapping[str, str]
    ) -> CellarRunCompounds: ...
