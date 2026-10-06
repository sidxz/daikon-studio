"""Register every existing protocol with Duar, so draft privacy covers what predates it.

The API runs this on every boot, in the background (`register_on_boot`); by hand:

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
import logging
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

logger = logging.getLogger(__name__)


@dataclass
class RegisterReport:
    registered: int = 0
    skipped: int = 0
    failures: list[str] = field(default_factory=list)


def _skipped(status: str) -> str:
    # A published protocol nobody can register stays invisible to members: say so loudly.
    return "PUBLISHED but hidden:" if status == ProtocolStatus.PUBLISHED.value else "skipped"


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
            log(f"{_skipped(status)} {label}: no creator recorded")
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
                log(f"{_skipped(status)} {label}: {error}")
            else:
                report.failures.append(f"{label}: {error}")
                log(f"FAILED {label}: {error}")
        except httpx.HTTPError as error:
            report.failures.append(f"{label}: {error}")
            log(f"FAILED {label}: {error}")
    return report


async def register_on_boot(duar: Any, session_factory: async_sessionmaker[AsyncSession]) -> None:
    """Best-effort, from the app lifespan: a failure is logged, never stops the API, and
    the next boot tries again.

    Skipped when the realm lookup failed (scope fell back to this service's own name),
    which is the CLI's whoami guard: registering then would file every protocol under
    the wrong service. A standalone deployment looks the same, so it registers by hand
    with `--standalone`.
    """
    # ponytail: walks every protocol on each boot (two Duar calls per published one);
    # fine at hundreds, add a "registered" column if it ever reaches the thousands.
    if duar.effective_scope == duar.service_name:
        logger.warning("protocol registration skipped: no realm scope (see the scope line)")
        return
    try:
        report = await register_all(session_factory, duar.permissions, log=logger.debug)
    except Exception:
        logger.exception("protocol registration failed")
        return
    log = logger.warning if report.failures else logger.info
    log(
        "protocols registered with Duar: %d registered, %d skipped, %d failed",
        report.registered,
        report.skipped,
        len(report.failures),
    )
    for failure in report.failures:
        logger.warning("protocol registration failed: %s", failure)


async def main() -> None:
    from daikonstudio.infrastructure.duar.auth import get_duar, log_effective_scope
    from daikonstudio.infrastructure.persistence.session import create_session_factory
    from daikonstudio.settings import Settings

    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    standalone = "--standalone" in sys.argv[1:]
    duar = get_duar()
    # Puts the SDK on the realm scope, as the app lifespan does. It returns None on any
    # failure, which would register every protocol under the wrong service name.
    whoami = await duar.fetch_whoami()
    print(f"Duar service scope: {duar.permissions.service_name}")
    log_effective_scope(duar)
    if whoami is None and not standalone:
        print("whoami failed; refusing to register under the wrong scope (--standalone to force)")
        raise SystemExit(1)
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
