"""What the agent may say about where the study stands.

A progress claim sits next to a decision, so it is kept only when the plan carries what it
cites. It is also dropped when it carries nothing: the page shows every step and every
criterion with its standing, and a turn that spends its claims saying a step has not run
yet has fewer left for what the result showed.
"""

from __future__ import annotations

import uuid
from typing import Any

from daikonstudio.application.studies.agent import check_progress
from daikonstudio.application.studies.schemas import ProgressClaim
from daikonstudio.domain.studies.study import SlotValue, Step, Study
from daikonstudio.infrastructure.tools.catalog import default_registry

REGISTRY = default_registry()


def _study(*tool_ids: str, **kw: Any) -> Study:
    steps = []
    for position, tool_id in enumerate(tool_ids):
        manifest = REGISTRY.get(tool_id)
        steps.append(
            Step(
                id=uuid.uuid4(),
                position=position,
                tool_id=tool_id,
                tool_version=manifest.version,
                name=manifest.name,
                purpose="p",
                params={},
                gated=False,
                always=False,
                expanded_from=None,
            )
        )
    return Study(
        workspace_id=uuid.uuid4(),
        created_by=uuid.uuid4(),
        goal="g",
        restated_goal="rg",
        checks=kw.pop("checks", ()),
        stop_conditions=kw.pop("stop_conditions", ()),
        steps=steps,
        **kw,
    )


def _claim(
    about: str, refers_to: list[int], label: str, cites: list[str], state: str = ""
) -> ProgressClaim:
    return ProgressClaim(about=about, refers_to=refers_to, label=label, cites=cites, state=state)


def test_a_claim_about_a_step_that_has_not_run_is_dropped() -> None:
    study = _study("fetch_pdb", "prepare_structure")
    study.start()
    step_one = sorted(study.steps, key=lambda s: s.position)[0]
    study.start_step(step_one.id)
    study.finish_step(
        step_one.id,
        [
            SlotValue(
                id=uuid.uuid4(),
                key="structure",
                kind="structure",
                blob_key="b/structure",
                summary={},
                produced_by=step_one.id,
                created_at=study.created_at,
            )
        ],
    )

    kept, dropped = check_progress(
        [
            _claim("step", [1], "Done: 4TRN fetched", ["Fetch a deposited structure"]),
            _claim("step", [2], "Still pending, gated", ["Prepare the structure"]),
        ],
        study,
    )

    assert [p.label for p in kept] == ["Done: 4TRN fetched"]
    assert dropped == 1


def test_a_claim_citing_words_the_plan_never_wrote_is_still_dropped() -> None:
    """The guard that makes a claim safe to print beside a result, unchanged."""
    study = _study("fetch_pdb", checks=("NAD is the only heterogen left.",))

    kept, dropped = check_progress(
        [
            _claim("check", [1], "met", ["NAD is the only heterogen left."]),
            _claim("check", [1], "met", ["a criterion nobody wrote"]),
        ],
        study,
    )

    assert [p.label for p in kept] == ["met"]
    assert dropped == 1


def test_the_declared_standing_is_carried_but_never_from_a_step() -> None:
    """The panel marks a criterion from this, so it is the agent's own word and not read
    out of its sentence. A step has no standing to report, whatever was put there."""
    study = _study("fetch_pdb", checks=("A candidate contains the catalytic residues.",))
    study.start()
    step_one = sorted(study.steps, key=lambda s: s.position)[0]
    study.start_step(step_one.id)
    study.finish_step(step_one.id, [])

    kept, dropped = check_progress(
        [
            _claim(
                "check",
                [1],
                "no candidate lists Tyr158",
                ["A candidate contains the catalytic residues."],
                state="not_met",
            ),
            _claim("step", [1], "Done", ["Fetch a deposited structure"], state="met"),
        ],
        study,
    )

    assert [(p.evidence, p.state) for p in kept] == [("check", "not_met"), ("step", "")]
    assert dropped == 0
