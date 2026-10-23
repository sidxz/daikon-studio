"""Pressing a proposal: the scientist's word, carried out by the edits the agent itself uses.

Each test offers a proposal the way the agent does, through `vet` so its steps are kept by id,
presses it, and reads what the study stored: the change on the step, the mark that stops a
second press, the event the scientist sees, and the run the loop was asked to start.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from returns.result import Result, Success

from daikonstudio.application.auth import AuthContext
from daikonstudio.application.studies.apply_proposal import ApplyCommand, ApplyProposal
from daikonstudio.application.studies.proposals import vet
from daikonstudio.application.studies.schemas import Proposal
from daikonstudio.domain.shared.errors import ConflictError, DomainError, ValidationError
from daikonstudio.domain.studies.study import Proposed, Step, Study, Turn
from daikonstudio.infrastructure.tools.catalog import default_registry
from tests.fakes.auth import FakeAuth
from tests.fakes.blob_store import InMemoryBlobStore
from tests.fakes.studies import InMemoryStudies

REGISTRY = default_registry()
AUTH = FakeAuth()
PDB = ("fetch_pdb", {"pdb_id": "1CA2"})
PROTEIN = (PDB, "prepare_structure", "find_pockets")


def _study(*plan: str | tuple[str, dict[str, str]]) -> Study:
    """The scientist's started study, its steps the tools named, in order."""
    steps = []
    for position, spec in enumerate(plan):
        tool_id, params = (spec, {}) if isinstance(spec, str) else spec
        manifest = REGISTRY.get(tool_id)
        steps.append(
            Step(
                id=uuid.uuid4(),
                position=position,
                tool_id=tool_id,
                tool_version=manifest.version,
                name=manifest.name,
                purpose="p",
                params=dict(params),
                gated=False,
                always=manifest.automatic,
                expanded_from=None,
            )
        )
    study = Study(
        workspace_id=AUTH.workspace_id,
        created_by=AUTH.user_id,
        goal="Find where a drug could bind carbonic anhydrase II.",
        restated_goal="",
        checks=(),
        stop_conditions=(),
        steps=steps,
    )
    study.start()
    return study


def _offer(study: Study, proposal: Proposal) -> Proposed:
    """The proposal as the agent leaves it in the thread: vetted, its steps named by id."""
    (offered,), _ = vet([proposal], study, REGISTRY)
    study.say(
        Turn(
            role="agent", kind="reply", say="Shall I?", proposals=(offered,), at=datetime.now(UTC)
        )
    )
    return offered


def _applied(study: Study, proposal: Proposed) -> bool:
    stored = study.proposal(proposal.id)
    return stored is not None and stored.applied_at is not None


class _Advance:
    """The run loop as a press calls it. It records the run it was asked to start and the kind
    of the last turn already in storage, because the real loop reads the study from there."""

    def __init__(self, studies: InMemoryStudies) -> None:
        self._studies = studies
        self.runs: list[tuple[uuid.UUID | None, uuid.UUID | None, str, str]] = []

    async def __call__(
        self,
        study_id: uuid.UUID,
        auth: AuthContext | None = None,
        *,
        start: uuid.UUID | None = None,
        until: uuid.UUID | None = None,
        started_by: str = "scientist",
    ) -> Result[Study, DomainError]:
        stored = self._studies.rows[study_id]
        self.runs.append((start, until, started_by, stored.thread.turns[-1].kind))
        return Success(stored)


def _service(study: Study) -> tuple[ApplyProposal, InMemoryStudies, _Advance]:
    studies = InMemoryStudies()
    studies.rows[study.id] = study
    advance = _Advance(studies)
    return ApplyProposal(studies, REGISTRY, InMemoryBlobStore(), advance), studies, advance


async def test_a_pressed_swap_changes_the_tool_and_leaves_its_problem_on_the_step() -> None:
    study = _study(*PROTEIN)
    fetch = study.steps[0]
    offered = _offer(
        study, Proposal(does="swap", step=1, tool_id="fetch_alphafold", why="No deposited entry.")
    )
    service, studies, advance = _service(study)

    result = await service(ApplyCommand(study.id, offered.id), AUTH)

    assert isinstance(result, Success)
    stored = studies.rows[study.id]
    step = stored.steps[0]
    assert (step.id, step.tool_id, step.params) == (fetch.id, "fetch_alphafold", {})
    # The new tool needs an accession the old one never took: the plan check says so on the step.
    assert "accession" in (step.problem or "")
    assert _applied(stored, offered)
    event = stored.thread.turns[-1]
    assert (event.role, event.kind, event.about) == ("event", "applied", fetch.id)
    assert event.say == (
        "Applied: Changed step 1 from Fetch a deposited structure to Fetch a predicted structure."
    )
    assert advance.runs == []  # an edit changes the plan and stops there


async def test_a_proposal_pressed_twice_is_made_once() -> None:
    study = _study(*PROTEIN)
    offered = _offer(study, Proposal(does="gate", step=3, gated=True, why="Look first."))
    service, studies, _ = _service(study)

    await service(ApplyCommand(study.id, offered.id), AUTH)
    again = await service(ApplyCommand(study.id, offered.id), AUTH)

    assert isinstance(again.failure(), ConflictError)
    assert again.failure().message == "This change has already been made."
    turns = studies.rows[study.id].thread.turns
    assert [turn.kind for turn in turns] == ["reply", "applied"]


async def test_a_pressed_run_starts_the_loop_at_the_steps_it_names_once_saved() -> None:
    study = _study(*PROTEIN)
    pockets = study.steps[2]
    offered = _offer(study, Proposal(does="run", step=3, until=3, why="Look again."))
    # As a proposal stored before an action's "none" was 0 or "" kept it: it still presses.
    offered.action.update(tool_id=None, purpose=None, gated=None, candidate=None)
    service, studies, advance = _service(study)

    result = await service(ApplyCommand(study.id, offered.id), AUTH)

    assert isinstance(result, Success)
    # By id, as the agent's own run is, and after the applied event was stored for the loop.
    assert advance.runs == [(pockets.id, pockets.id, "agent", "applied")]
    stored = studies.rows[study.id]
    assert _applied(stored, offered)
    event = stored.thread.turns[-1]
    assert (event.kind, event.about) == ("applied", pockets.id)
    assert event.say == "Applied: Started a run of step 3 (Find binding pockets)."


async def test_a_run_with_nothing_left_to_run_is_refused_and_stays_unapplied() -> None:
    study = _study(*PROTEIN)
    for step in study.steps:
        study.start_step(step.id)
        study.finish_step(step.id, [])
    offered = _offer(study, Proposal(does="run", why="Go on."))
    service, studies, advance = _service(study)

    result = await service(ApplyCommand(study.id, offered.id), AUTH)

    assert isinstance(result.failure(), ValidationError)
    assert result.failure().message == "There is nothing left to run."
    assert advance.runs == []
    stored = studies.rows[study.id]
    assert not _applied(stored, offered)
    assert [turn.kind for turn in stored.thread.turns] == ["reply"]
