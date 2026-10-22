"""What the agent may offer the scientist: only what the study could carry out as it stands.

A proposal is a button. One that pressing could not carry out is worse than none, so each is
checked as the edit it would make is checked, and is kept with its steps named by id.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from daikonstudio.application.planner.plan import Setting
from daikonstudio.application.studies.proposals import vet
from daikonstudio.application.studies.schemas import Proposal
from daikonstudio.domain.studies.study import SlotValue, Step, Study
from daikonstudio.infrastructure.tools.catalog import default_registry

REGISTRY = default_registry()
PDB = ("fetch_pdb", {"pdb_id": "1CA2"})


def _study(*plan: str | tuple[str, dict[str, str]]) -> Study:
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
    return Study(
        workspace_id=uuid.uuid4(),
        created_by=uuid.uuid4(),
        goal="g",
        restated_goal="rg",
        checks=(),
        stop_conditions=(),
        steps=steps,
    )


def test_vet_drops_a_swap_outside_the_alternatives() -> None:
    """Fetching a predicted model takes the deposited one's place; finding pockets does not."""
    study = _study(PDB, "prepare_structure", "find_pockets")

    kept, dropped = vet(
        [
            Proposal(does="swap", step=1, tool_id="find_pockets", why="w"),
            Proposal(does="swap", step=1, tool_id="fetch_alphafold", why="w"),
        ],
        study,
        REGISTRY,
    )

    assert [p.action["tool_id"] for p in kept] == ["fetch_alphafold"]
    assert dropped == 1


def test_vet_drops_an_insert_whose_inputs_nothing_holds() -> None:
    study = _study(PDB)
    insert = Proposal(does="insert", tool_id="find_pockets", why="Look for a site.")

    assert vet([insert], study, REGISTRY) == ([], 1)

    study.bring(
        SlotValue(
            id=uuid.uuid4(),
            key="structure",
            kind="structure",
            blob_key="b",
            summary={},
            produced_by=None,
            created_at=datetime.now(UTC),
        )
    )
    kept, dropped = vet([insert], study, REGISTRY)
    assert ([p.action["tool_id"] for p in kept], dropped) == (["find_pockets"], 0)


def test_vet_keeps_a_rerun_with_valid_settings_and_resolves_the_step_id() -> None:
    study = _study(PDB, "prepare_structure", "find_pockets")
    prepare = study.steps[1]

    kept, dropped = vet(
        [
            Proposal(
                does="settings",
                step=2,
                settings=[Setting(name="keep_ligands", value="")],
                why="Strip the zinc and look again.",
            ),
            # A name the tool does not take would be silently ignored when it ran.
            Proposal(
                does="settings", step=2, settings=[Setting(name="strip", value="ZN")], why="w"
            ),
        ],
        study,
        REGISTRY,
    )

    assert dropped == 1
    assert [(p.action, p.why) for p in kept] == [
        (
            {
                "does": "settings",
                "step_id": str(prepare.id),
                "tool_id": None,
                "before_id": None,
                "purpose": None,
                "settings": {"keep_ligands": ""},
                "gated": None,
                "candidate": None,
                "until_id": None,
            },
            "Strip the zinc and look again.",
        )
    ]
