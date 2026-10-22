"""Opening a study, and attaching files to it.

A study opens on the scientist's first message, with whatever files they have to hand, and the
agent asks before anything is planned. Each file becomes a value the scientist brought, kept
once and by kind: inputs bind to a value by its kind and the newest wins, so a second file of
a kind would quietly shadow the first, and is refused instead. Nothing is half done. A file
that is refused is refused before it is kept, and a refusal part-way through a batch takes back
what the files before it stored, so no study, no value and no blob is left behind.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePath
from typing import Any

import polars as pl
from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.study_repository import StudyRepository
from daikonstudio.application.ports.table_profiler import TableProfile, TableProfiler
from daikonstudio.application.studies.advance_study import in_flight
from daikonstudio.application.studies.agent import StudyAgent
from daikonstudio.application.studies.edits import recheck, replace_automatic
from daikonstudio.application.studies.get_study import GetStudy
from daikonstudio.application.studies.keys import value_key
from daikonstudio.application.tools import codecs
from daikonstudio.application.tools.manifest import SlotKind
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.application.tools.structures import describe_pdb
from daikonstudio.domain.shared.errors import ConflictError, DomainError, ValidationError
from daikonstudio.domain.studies.study import SlotValue, Study, Turn

RUNNING = "A run is under way in this study. Wait for it to stop, then try again."

# A goal shorter than this is not yet something to ask questions about.
_MIN_GOAL = 15
_BOM = b"\xef\xbb\xbf"
_KINDS = {
    ".csv": SlotKind.COMPOUND_TABLE.value,
    ".tsv": SlotKind.COMPOUND_TABLE.value,
    ".tab": SlotKind.COMPOUND_TABLE.value,
    ".pdb": SlotKind.STRUCTURE.value,
}


@dataclass(frozen=True)
class Upload:
    filename: str
    content: bytes


@dataclass(frozen=True)
class OpenCommand:
    goal: str
    provider: str | None = None
    files: tuple[Upload, ...] = ()


def kind_of(filename: str) -> str:
    """The kind of value a file becomes, by its extension.

    mmCIF is named in the refusal because it is the one a structural biologist reaches for
    next, and nothing in the backend reads it yet: accepted, it would only fail later, in the
    step that wanted it.
    """
    kind = _KINDS.get(PurePath(filename).suffix.lower())
    if kind is None:
        raise ValidationError(
            "That file type is not supported.",
            detail="Attach a .csv, .tsv or .pdb file; .cif is not read yet.",
        )
    return kind


def _spelled(kind: str) -> str:
    return kind.replace("_", " ")


def _plain(profile: TableProfile) -> dict[str, Any]:
    """The profile as data the study row can hold.

    JSON has no NaN and no infinity, and the profiler reads "NaN" and "inf" in a column of text
    as numbers. A range that is not finite is no range, so it is left empty rather than stored.
    """
    data = dataclasses.asdict(profile)
    for column in data["columns"]:
        stats = (column["minimum"], column["maximum"], column["median"])
        if not all(stat is None or math.isfinite(stat) for stat in stats):
            column.update(minimum=None, maximum=None, median=None)
    return data


async def bring(
    study: Study, upload: Upload, *, store: BlobStore, profiler: TableProfiler
) -> SlotValue:
    """Read a file into the study as a value the scientist brought, and keep it.

    A table is stored as parquet with its profile beside it in the summary, which is what the
    agent is shown at planning time; a structure is stored in the shape `fetch_pdb` makes, with
    the `meta` the file itself can give. The key is the kind, since there is only one of each.
    """
    kind = kind_of(upload.filename)
    if any(v.kind == kind and v.produced_by is None for v in study.values):
        raise ValidationError(f"A study takes one {_spelled(kind)} file.")
    if kind == SlotKind.COMPOUND_TABLE:
        raw = upload.content.removeprefix(_BOM)
        tabbed = upload.filename.lower().endswith((".tsv", ".tab"))
        try:
            frame = pl.read_csv(
                io.BytesIO(raw), separator="\t" if tabbed else ",", encoding="utf8-lossy"
            )
        except pl.exceptions.PolarsError as error:
            raise ValidationError(
                "The compounds file could not be read.", detail=str(error)
            ) from error
        blob, summary = codecs.encode(kind, frame)
        summary["profile"] = _plain(profiler.profile(raw, filename=upload.filename))
    else:
        text = upload.content.decode("utf-8", errors="replace")
        try:
            meta = describe_pdb(text)
        except ValueError as error:
            raise ValidationError(
                "The structure file could not be read.", detail=str(error)
            ) from error
        blob, summary = codecs.encode(kind, {"format": "pdb", "text": text, "meta": meta})
    value_id = uuid.uuid4()
    key = value_key(study.workspace_id, study.id, value_id, codecs.extension(kind))
    await asyncio.to_thread(store.put_bytes, key, blob)
    value = SlotValue(
        id=value_id,
        key=kind,
        kind=kind,
        blob_key=key,
        summary=summary,
        produced_by=None,
        created_at=datetime.now(UTC),
    )
    study.bring(value)
    return value


async def _bring_all(
    study: Study, files: Sequence[Upload], *, store: BlobStore, profiler: TableProfiler
) -> list[SlotValue]:
    """Bring every file or none: a refusal takes back the blobs of the files before it."""
    brought: list[SlotValue] = []
    try:
        for upload in files:
            brought.append(await bring(study, upload, store=store, profiler=profiler))
    except Exception:
        for value in brought:
            await asyncio.to_thread(store.delete, value.blob_key)
        raise
    return brought


class OpenStudy:
    """The first message: a draft with the files it came with, and the agent's first turn."""

    def __init__(
        self,
        studies: StudyRepository,
        store: BlobStore,
        profiler: TableProfiler,
        agent: StudyAgent,
    ) -> None:
        self._studies = studies
        self._store = store
        self._profiler = profiler
        self._agent = agent

    async def __call__(
        self, command: OpenCommand, auth: AuthContext | None = None
    ) -> Result[Study, DomainError]:
        """Open the draft and take the agent's first turn on it.

        The draft is kept before the agent is asked, so a provider that fails leaves a study
        with no agent turn yet: its phase is still asking, and the next message asks again.
        """
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        goal = command.goal.strip()
        if len(goal) < _MIN_GOAL:
            return Failure(ValidationError("Say what you want to find out."))
        study = Study(
            workspace_id=auth.workspace_id,
            created_by=auth.user_id,
            goal=goal,
            restated_goal="",
            checks=(),
            stop_conditions=(),
            steps=[],
        )
        try:
            await _bring_all(study, command.files, store=self._store, profiler=self._profiler)
        except ValidationError as error:
            return Failure(error)
        await self._studies.add(study)
        return Success(await self._agent.ask(study, command.provider))


class AttachFiles:
    """More files for a study that already exists, whatever its phase."""

    def __init__(
        self,
        studies: StudyRepository,
        store: BlobStore,
        profiler: TableProfiler,
        tools: ToolRegistry,
    ) -> None:
        self._studies = studies
        self._get = GetStudy(studies)
        self._store = store
        self._profiler = profiler
        self._tools = tools

    async def __call__(
        self, study_id: uuid.UUID, files: tuple[Upload, ...], auth: AuthContext | None = None
    ) -> Result[Study, DomainError]:
        """Bring the files, then match the plan to them.

        A plan drafted before its file is waiting on it: the file makes the automatic steps
        that prepare it possible, and settles the problem that it was missing. So they are
        placed and the plan is checked again, as they are after any edit.
        """
        require_authenticated(auth)
        require_editor(auth)
        found = await self._get(study_id, auth)
        if not isinstance(found, Success):
            return Failure(found.failure())
        study = found.unwrap()
        if in_flight(study.id):
            return Failure(ConflictError(RUNNING))
        try:
            brought = await _bring_all(study, files, store=self._store, profiler=self._profiler)
        except ValidationError as error:
            return Failure(error)
        replace_automatic(study, self._tools)
        recheck(study, self._tools)
        for upload, value in zip(files, brought, strict=True):
            say = f"Attached {upload.filename} as {_spelled(value.kind)}."
            study.say(Turn(role="event", kind="attached", say=say, at=datetime.now(UTC)))
        await self._studies.update(study)
        return Success(study)
