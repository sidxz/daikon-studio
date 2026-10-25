"""SQLAlchemy Study repository.

Steps, values and the thread live as JSON on the study row: datetimes as ISO strings,
uuids as strings. `update` is optimistic on `version`, like the Run repository.
"""

from __future__ import annotations

import builtins
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CursorResult, select
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from daikonstudio.domain.shared.errors import ConcurrencyConflictError
from daikonstudio.domain.studies.study import (
    Acted,
    Answered,
    Asked,
    Pointer,
    Proposed,
    SlotValue,
    Step,
    StepStatus,
    Study,
    Thread,
    Turn,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.studies.models import StudyModel


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _when(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _id(value: uuid.UUID | None) -> str | None:
    return str(value) if value else None


def _uuid(value: str | None) -> uuid.UUID | None:
    return uuid.UUID(value) if value else None


def _thread_to_json(t: Thread) -> dict[str, Any]:
    return {
        "summary": t.summary,
        "turns": [
            {
                "role": turn.role,
                "kind": turn.kind,
                "say": turn.say,
                "about": _id(turn.about),
                "questions": [
                    {"ask": q.ask, "why": q.why, "options": list(q.options), "default": q.default}
                    for q in turn.questions
                ],
                "points": [
                    {
                        "step": _id(p.step),
                        "at": p.at,
                        "label": p.label,
                        "evidence": p.evidence,
                        "cites": list(p.cites),
                    }
                    for p in turn.points
                ],
                "proposals": [
                    {
                        "id": str(p.id),
                        "action": p.action,
                        "why": p.why,
                        "applied_at": _iso(p.applied_at),
                    }
                    for p in turn.proposals
                ],
                "unresolved": list(turn.unresolved),
                "actions": [
                    {"does": a.does, "step": _id(a.step), "detail": a.detail, "quoting": a.quoting}
                    for a in turn.actions
                ],
                "looked": list(turn.looked),
                "wanted": list(turn.wanted),
                "target_kinds": list(turn.target_kinds),
                "dropped": turn.dropped,
                "silent": turn.silent,
                "written_by": turn.written_by,
                "at": _iso(turn.at),
            }
            for turn in t.turns
        ],
    }


def _thread_from_json(d: dict[str, Any]) -> Thread:
    # `{}` -- the column's default -- is an empty thread.
    return Thread(
        summary=d.get("summary", ""),
        turns=[
            Turn(
                role=t["role"],
                kind=t.get("kind", ""),
                say=t["say"],
                about=_uuid(t.get("about")),
                questions=tuple(
                    Asked(
                        ask=q["ask"],
                        why=q["why"],
                        options=tuple(q.get("options", ())),
                        default=q.get("default"),
                    )
                    for q in t.get("questions", [])
                ),
                points=tuple(
                    Pointer(
                        step=_uuid(p["step"]),
                        at=p["at"],
                        label=p["label"],
                        evidence=p["evidence"],
                        cites=tuple(p.get("cites", ())),
                    )
                    for p in t.get("points", [])
                ),
                proposals=tuple(
                    Proposed(
                        id=uuid.UUID(p["id"]),
                        action=p["action"],
                        why=p["why"],
                        applied_at=_when(p.get("applied_at")),
                    )
                    for p in t.get("proposals", [])
                ),
                unresolved=tuple(t.get("unresolved", ())),
                actions=tuple(
                    Acted(
                        does=a["does"],
                        step=_uuid(a["step"]),
                        detail=a["detail"],
                        quoting=a["quoting"],
                    )
                    for a in t.get("actions", [])
                ),
                looked=tuple(t.get("looked", ())),
                wanted=tuple(t.get("wanted", ())),
                target_kinds=tuple(t.get("target_kinds", ())),
                dropped=t.get("dropped", 0),
                silent=t.get("silent", False),
                written_by=t.get("written_by"),
                at=datetime.fromisoformat(t["at"]),
            )
            for t in d.get("turns", [])
        ],
    )


def _step_to_json(s: Step) -> dict[str, Any]:
    return {
        "id": str(s.id),
        "position": s.position,
        "tool_id": s.tool_id,
        "tool_version": s.tool_version,
        "name": s.name,
        "purpose": s.purpose,
        "params": s.params,
        "gated": s.gated,
        "always": s.always,
        "expanded_from": s.expanded_from,
        "status": s.status.value,
        "error": s.error,
        "problem": s.problem,
        "started_at": _iso(s.started_at),
        "finished_at": _iso(s.finished_at),
    }


def _step_from_json(d: dict[str, Any]) -> Step:
    return Step(
        id=uuid.UUID(d["id"]),
        position=d["position"],
        tool_id=d["tool_id"],
        tool_version=d["tool_version"],
        name=d["name"],
        purpose=d["purpose"],
        params=d["params"],
        gated=d["gated"],
        always=d["always"],
        expanded_from=d["expanded_from"],
        status=StepStatus(d["status"]),
        error=d["error"],
        problem=d["problem"],
        started_at=_when(d["started_at"]),
        finished_at=_when(d["finished_at"]),
    )


def _value_to_json(v: SlotValue) -> dict[str, Any]:
    return {
        "id": str(v.id),
        "key": v.key,
        "kind": v.kind,
        "blob_key": v.blob_key,
        "summary": v.summary,
        "produced_by": _id(v.produced_by),
        "asserted_by": _id(v.asserted_by),
        "created_at": v.created_at.isoformat(),
    }


def _value_from_json(d: dict[str, Any]) -> SlotValue:
    return SlotValue(
        id=uuid.UUID(d["id"]),
        key=d["key"],
        kind=d["kind"],
        blob_key=d["blob_key"],
        summary=d["summary"],
        produced_by=_uuid(d["produced_by"]),
        asserted_by=_uuid(d["asserted_by"]),
        created_at=datetime.fromisoformat(d["created_at"]),
    )


def _to_domain(model: StudyModel) -> Study:
    return Study(
        id=model.id,
        workspace_id=model.workspace_id,
        created_by=model.created_by,
        goal=model.goal,
        restated_goal=model.restated_goal,
        checks=tuple(model.checks),
        stop_conditions=tuple(model.stop_conditions),
        steps=[_step_from_json(s) for s in model.steps],
        values=[_value_from_json(v) for v in model.values],
        answers=[Answered(ask=a["ask"], answer=a["answer"]) for a in model.answers],
        thread=_thread_from_json(model.thread),
        started_at=model.started_at,
        mode=model.mode,
        amended_after_steps=model.amended_after_steps,
        created_at=model.created_at,
        updated_at=model.updated_at,
        version=model.version,
    )


def _to_model(study: Study) -> StudyModel:
    return StudyModel(
        id=study.id,
        workspace_id=study.workspace_id,
        created_by=study.created_by,
        goal=study.goal,
        restated_goal=study.restated_goal,
        checks=list(study.checks),
        stop_conditions=list(study.stop_conditions),
        steps=[_step_to_json(s) for s in study.steps],
        values=[_value_to_json(v) for v in study.values],
        answers=[{"ask": a.ask, "answer": a.answer} for a in study.answers],
        thread=_thread_to_json(study.thread),
        started_at=study.started_at,
        mode=study.mode,
        amended_after_steps=study.amended_after_steps,
        created_at=study.created_at,
        updated_at=study.updated_at,
        version=study.version,
    )


class SqlAlchemyStudyRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(self, study: Study) -> None:
        async with self._sessions() as session:
            session.add(_to_model(study))
            await session.commit()

    async def update(self, study: Study) -> None:
        model = _to_model(study)
        expected = study.version
        async with self._sessions() as session:
            result = await session.execute(
                sa_update(StudyModel)
                .where(
                    StudyModel.id == study.id,
                    StudyModel.workspace_id == study.workspace_id,
                    StudyModel.version == expected,
                )
                .values(
                    steps=model.steps,
                    values=model.values,
                    thread=model.thread,
                    restated_goal=model.restated_goal,
                    checks=model.checks,
                    stop_conditions=model.stop_conditions,
                    answers=model.answers,
                    started_at=model.started_at,
                    amended_after_steps=model.amended_after_steps,
                    updated_at=model.updated_at,
                    version=expected + 1,
                )
            )
            assert isinstance(result, CursorResult)
            if result.rowcount == 0:
                await session.rollback()
                raise ConcurrencyConflictError("Study", str(study.id))
            await session.commit()
        study.version = expected + 1

    async def get(self, workspace_id: uuid.UUID, study_id: uuid.UUID) -> Study | None:
        async with self._sessions() as session:
            model = (
                await session.execute(
                    select(StudyModel).where(
                        StudyModel.workspace_id == workspace_id, StudyModel.id == study_id
                    )
                )
            ).scalar_one_or_none()
            return _to_domain(model) if model else None

    async def list(
        self, workspace_id: uuid.UUID, *, created_by: uuid.UUID | None = None, limit: int = 50
    ) -> builtins.list[Study]:
        query = select(StudyModel).where(StudyModel.workspace_id == workspace_id)
        if created_by is not None:
            query = query.where(StudyModel.created_by == created_by)
        async with self._sessions() as session:
            rows = (
                await session.execute(query.order_by(StudyModel.created_at.desc()).limit(limit))
            ).scalars()
            return [_to_domain(m) for m in rows]
