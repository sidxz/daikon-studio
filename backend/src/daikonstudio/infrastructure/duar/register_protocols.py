"""Register every existing protocol with Duar, so draft privacy covers what predates it.

    make register-protocols     # or: python -m daikonstudio.infrastructure.duar.register_protocols

New protocols register themselves on create (`AccessControlledProtocolRepository`);
this covers the ones written before. A draft becomes private to its creator, a
published one workspace-wide. It walks every workspace, so it reads the table
directly rather than through the workspace-scoped repository.

Idempotent: Duar's register is a no-op on an existing resource and never changes its
visibility, so a published protocol is also pushed to "workspace" explicitly (an empty
token sends only the service key). Re-running after a partial failure is safe. Duar's
permission cache in the API may lag by up to 120 s. One line per protocol; exit 1 if
anything failed.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from duar_auth import DuarError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from daikonstudio.domain.catalog.protocol import ProtocolStatus
from daikonstudio.infrastructure.duar.protocol_access import RESOURCE_TYPE
from daikonstudio.infrastructure.persistence.sqlalchemy.catalog.models import (
    InSilicoProtocolModel,
)


@dataclass
class RegisterReport:
    registered: int = 0
    skipped: int = 0
    failures: list[str] = field(default_factory=list)


async def register_all(
    session_factory: async_sessionmaker[AsyncSession],
    permissions: Any,
    log: Callable[[str], None] = print,
) -> RegisterReport:
    report = RegisterReport()
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(
                    InSilicoProtocolModel.id,
                    InSilicoProtocolModel.workspace_id,
                    InSilicoProtocolModel.created_by,
                    InSilicoProtocolModel.status,
                    InSilicoProtocolModel.name,
                ).order_by(InSilicoProtocolModel.created_at)
            )
        ).all()

    for protocol_id, workspace_id, created_by, status, name in rows:
        label = f"protocol {protocol_id} ({name})"
        if created_by is None:
            report.skipped += 1
            log(f"skipped {label}: no creator recorded")
            continue
        published = status == ProtocolStatus.PUBLISHED.value
        try:
            await permissions.register_resource(
                resource_type=RESOURCE_TYPE,
                resource_id=protocol_id,
                workspace_id=workspace_id,
                owner_id=created_by,
                visibility="workspace" if published else "private",
            )
            if published:
                await permissions.update_visibility("", RESOURCE_TYPE, protocol_id, "workspace")
            report.registered += 1
            log(f"registered {label}: {'workspace' if published else 'private'}")
        except DuarError as error:
            if error.status_code is not None and error.status_code < 500:
                report.skipped += 1
                log(f"skipped {label}: {error}")
            else:
                report.failures.append(f"{label}: {error}")
                log(f"FAILED {label}: {error}")
        except httpx.HTTPError as error:
            report.failures.append(f"{label}: {error}")
            log(f"FAILED {label}: {error}")
    return report


async def main() -> None:
    from daikonstudio.infrastructure.duar.auth import get_duar
    from daikonstudio.infrastructure.persistence.session import create_session_factory
    from daikonstudio.settings import Settings

    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    duar = get_duar()
    await duar.fetch_whoami()  # puts the SDK on the realm scope, as the app lifespan does
    report = await register_all(session_factory, duar.permissions)
    print(
        f"{report.registered} registered, {report.skipped} skipped, {len(report.failures)} failed"
    )
    if report.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
