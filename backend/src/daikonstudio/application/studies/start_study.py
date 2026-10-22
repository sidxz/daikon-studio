"""Starting a study: a plan the scientist approved, saved with what they brought."""

from __future__ import annotations

import asyncio
import io
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import polars as pl
from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.planner.plan import (
    DraftPlan,
    PlannedStep,
    _handed_over,
    incoherences,
)
from daikonstudio.application.planner.plan import Step as PlanStep
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.study_repository import StudyRepository
from daikonstudio.application.studies.keys import value_key
from daikonstudio.application.tools import codecs
from daikonstudio.application.tools.manifest import SlotKind
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.shared.errors import DomainError, ValidationError
from daikonstudio.domain.studies.study import Answered, SlotValue, Step, Study


@dataclass(frozen=True, kw_only=True)
class StepSpec:
    tool_id: str
    purpose: str
    params: dict[str, Any]
    gated: bool
    always: bool = False
    expanded_from: str | None = None


@dataclass(frozen=True, kw_only=True)
class StartStudyCommand:
    goal: str
    restated_goal: str
    checks: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    steps: tuple[StepSpec, ...]
    compounds: bytes | None = None
    filename: str | None = None
    answers: tuple[Answered, ...] = ()
    planning: Mapping[str, Any] = field(default_factory=dict)


class StartStudy:
    def __init__(self, studies: StudyRepository, tools: ToolRegistry, store: BlobStore) -> None:
        self._studies = studies
        self._tools = tools
        self._store = store

    async def __call__(
        self, command: StartStudyCommand, auth: AuthContext | None = None
    ) -> Result[Study, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        if not command.steps:
            return Failure(ValidationError("A study needs at least one step."))
        planned = [
            PlannedStep(
                tool_id=s.tool_id,
                name=s.tool_id,
                purpose=s.purpose,
                gated=s.gated,
                params=s.params,
                expanded_from=s.expanded_from,
                always=s.always,
            )
            for s in command.steps
        ]
        if not command.compounds:
            asked = DraftPlan(
                restated_goal=command.restated_goal,
                steps=[
                    PlanStep(tool_id=s.tool_id, purpose=s.purpose, gated=s.gated) for s in planned
                ],
            )
            if SlotKind.COMPOUND_TABLE in _handed_over(asked, self._tools):
                return Failure(
                    ValidationError(
                        "Attach the compounds file: this plan works on a compound table "
                        "and none was sent."
                    )
                )
        problems = incoherences(
            planned,
            self._tools,
            starting_with={SlotKind.COMPOUND_TABLE} if command.compounds else set(),
        )
        if problems:
            return Failure(ValidationError("; ".join(p.message for p in problems)))

        steps = []
        for position, spec in enumerate(command.steps):
            manifest = self._tools.get(spec.tool_id)
            steps.append(
                Step(
                    position=position,
                    tool_id=spec.tool_id,
                    tool_version=manifest.version,
                    name=manifest.name,
                    purpose=spec.purpose,
                    params=spec.params,
                    gated=spec.gated,
                    always=spec.always,
                    expanded_from=spec.expanded_from,
                )
            )
        study = Study(
            workspace_id=auth.workspace_id,
            created_by=auth.user_id,
            goal=command.goal,
            restated_goal=command.restated_goal,
            checks=command.checks,
            stop_conditions=command.stop_conditions,
            steps=steps,
            answers=command.answers,
            planning=command.planning,
        )

        if command.compounds:
            raw = command.compounds.removeprefix(b"\xef\xbb\xbf")
            tabbed = (command.filename or "").lower().endswith((".tsv", ".tab"))
            try:
                frame = pl.read_csv(
                    io.BytesIO(raw), separator="\t" if tabbed else ",", encoding="utf8-lossy"
                )
            except pl.exceptions.PolarsError as error:
                return Failure(
                    ValidationError("The compounds file could not be read.", detail=str(error))
                )
            blob, summary = codecs.encode("compound_table", frame)
            value_id = uuid.uuid4()
            key = value_key(auth.workspace_id, study.id, value_id, "parquet")
            await asyncio.to_thread(self._store.put_bytes, key, blob)
            study.bring(
                SlotValue(
                    id=value_id,
                    key="compounds",
                    kind="compound_table",
                    blob_key=key,
                    summary=summary,
                    produced_by=None,
                    created_at=datetime.now(UTC),
                )
            )

        await self._studies.add(study)
        return Success(study)
