"""Building a dataset in the background, so the wizard can show progress.

`StartDatasetBuild` records a DatasetBuild, returns it at once, and runs the same
`CreateDataset` on a task of its own: the RDKit work goes to a worker thread, and the
task writes the build's stage and row count every second until it ends. The wizard
polls `GetDatasetBuild`. The synchronous `POST /datasets` stays for scripts.

ponytail: the build runs inside the API process, so a restart loses it; the row then
goes stale and reads as interrupted (see DatasetBuild.is_stale). Move it to a runner
job if builds ever need to survive a deploy.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import (
    AuthContext,
    require_authenticated,
    require_editor,
    require_same_workspace,
)
from daikonstudio.application.data.create_dataset import (
    BuildProgress,
    CreateDataset,
    CreateDatasetCommand,
)
from daikonstudio.application.ports.dataset_build_repository import DatasetBuildRepository
from daikonstudio.domain.data.dataset_build import INTERRUPTED, DatasetBuild
from daikonstudio.domain.shared.errors import DomainError, NotFoundError

logger = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 1.0

# asyncio holds only weak references to tasks: without these, a build could be
# garbage-collected mid-run.
_running: set[asyncio.Task[None]] = set()

_CRASHED = {
    "error": "InternalError",
    "message": "The dataset could not be built because of an unexpected server error.",
}


@dataclass(frozen=True)
class _Auth:
    """The caller's identity, copied out of the request: the build outlives it."""

    user_id: uuid.UUID
    workspace_id: uuid.UUID
    workspace_role: str


class StartDatasetBuild:
    def __init__(self, builds: DatasetBuildRepository, create: CreateDataset) -> None:
        self._builds = builds
        self._create = create

    async def __call__(
        self, command: CreateDatasetCommand, auth: AuthContext | None = None
    ) -> Result[DatasetBuild, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        build = DatasetBuild(
            workspace_id=auth.workspace_id, created_by=auth.user_id, name=command.name
        )
        await self._builds.add(build)
        caller = _Auth(auth.user_id, auth.workspace_id, auth.workspace_role)
        task = asyncio.create_task(self._run(build, command, caller))
        _running.add(task)
        task.add_done_callback(_running.discard)
        return Success(build)

    async def _run(self, build: DatasetBuild, command: CreateDatasetCommand, auth: _Auth) -> None:
        progress = BuildProgress()
        work = asyncio.ensure_future(self._create(command, auth=auth, progress=progress))
        try:
            while not work.done():
                await asyncio.wait({work}, timeout=HEARTBEAT_SECONDS)
                if not work.done():
                    build.report(progress.stage, progress.done, progress.total)
                    await self._save(build)
            result = work.result()
        except Exception:
            logger.exception("Dataset build %s crashed", build.id)
            build.fail(dict(_CRASHED))
        else:
            if isinstance(result, Failure):
                build.fail(result.failure().to_body())
            else:
                build.succeed(result.unwrap().id)
        await self._save(build)

    async def _save(self, build: DatasetBuild) -> None:
        # A failed progress write must not kill the build doing the real work. A
        # failed final write leaves it running until it reads as stale.
        try:
            await self._builds.save(build)
        except Exception:
            logger.exception("Saving dataset build %s failed", build.id)


class GetDatasetBuild:
    def __init__(self, builds: DatasetBuildRepository) -> None:
        self._builds = builds

    async def __call__(
        self, build_id: uuid.UUID, auth: AuthContext | None = None
    ) -> Result[DatasetBuild, DomainError]:
        require_authenticated(auth)
        build = await self._builds.get(build_id)
        if build is None:
            return Failure(NotFoundError("Dataset build", str(build_id)))
        require_same_workspace(auth, build.workspace_id, entity_type="Dataset build")
        if build.is_stale(datetime.now(UTC)):
            build.fail(dict(INTERRUPTED))
            await self._builds.save(build)
        return Success(build)
