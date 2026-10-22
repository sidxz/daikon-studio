"""The agent at a stage may only say what its step's result supports.

Everything here is about the check, because the check is the only reason it is
safe to put a model next to a result a scientist is about to act on. A claim that
reads well and cites a residue the pocket does not have is exactly the failure
this exists to catch, and it is the one a reader cannot catch for themselves.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from daikonstudio.application.studies.stage_chat import CHECKERS, Claim, _check_pockets
from daikonstudio.domain.studies.study import Turn


def _pocket(rank: int, residues: list[tuple[str, int]], **extra: Any) -> dict[str, Any]:
    return {
        "rank": rank,
        "residues": [{"chain": "A", "resname": n, "resseq": s} for n, s in residues],
        "ligand_overlap": extra.get("ligand_overlap", []),
        "annotated": extra.get("annotated", []),
        "interface": extra.get("interface", False),
    }


DATA = {
    "meta": {},
    "pockets": [
        _pocket(1, [("ALA", 128), ("LYS", 132)]),
        _pocket(
            10,
            [("HIS", 94), ("HIS", 96), ("THR", 200)],
            ligand_overlap=[{"comp_id": "ZN", "shared": 3, "of": 3, "additive": False}],
        ),
        _pocket(11, [("ASP", 25), ("ILE", 50)], interface=True),
    ],
}


def _claim(**kw: Any) -> Claim:
    return Claim(
        candidates=kw.get("candidates", [10]),
        label=kw.get("label", "the zinc cone"),
        evidence=kw.get("evidence", "lining_residues"),
        cites=kw.get("cites", ["HIS94", "HIS96"]),
    )


def test_a_grounded_claim_survives_and_reads_back() -> None:
    points, dropped = _check_pockets([_claim()], DATA)
    assert dropped == 0
    assert points[0].at == "rank 10"
    assert points[0].label == "the zinc cone"
    assert points[0].cites == ("HIS94", "HIS96")


def test_a_residue_the_candidate_does_not_have_costs_the_claim() -> None:
    """The whole claim goes, not just the bad residue: a reading with an invented
    residue in it was not read off this result."""
    points, dropped = _check_pockets([_claim(cites=["HIS94", "TYR158"])], DATA)
    assert points == []
    assert dropped == 1


def test_residues_are_matched_however_they_are_spelled() -> None:
    points, _ = _check_pockets([_claim(cites=["A:His 94", "his96"])], DATA)
    assert len(points) == 1


def test_a_rank_that_is_not_there_costs_the_claim() -> None:
    points, dropped = _check_pockets([_claim(candidates=[10, 99])], DATA)
    assert points == []
    assert dropped == 1


def test_a_claim_with_nothing_cited_is_an_opinion_and_is_dropped() -> None:
    points, dropped = _check_pockets([_claim(cites=[])], DATA)
    assert points == []
    assert dropped == 1


def test_evidence_must_be_in_the_result_not_just_asserted() -> None:
    """Calling candidate 10 an interface pocket is a claim about a column, and
    that column says no."""
    points, dropped = _check_pockets([_claim(evidence="interface")], DATA)
    assert points == []
    assert dropped == 1
    kept, _ = _check_pockets(
        [_claim(candidates=[11], cites=["ASP25"], evidence="interface")], DATA
    )
    assert len(kept) == 1


def test_one_claim_may_join_candidates_the_tool_reported_apart() -> None:
    """The fragmentation case: residues are checked against the candidates
    together, because the point of the claim is that they are one site."""
    points, dropped = _check_pockets(
        [_claim(candidates=[1, 10], cites=["ALA128", "THR200"], label="one cleft")], DATA
    )
    assert dropped == 0
    assert points[0].at == "ranks 1, 10"


def test_a_kind_with_no_checker_is_not_discussed() -> None:
    """There is no permissive default on purpose: a result nobody can check is a
    result the agent has nothing to be held to. `stage_chat` refuses on
    `kind not in CHECKERS`, so an unchecked kind has to be absent rather than
    fall back to something lenient. Stated as the two properties that matter, so
    that adding a second checker is not a test failure."""
    assert "structure" not in CHECKERS
    assert CHECKERS["pocket_candidates"] is _check_pockets


class TestThreads:
    def test_a_turn_starts_the_thread_and_later_turns_join_it(self) -> None:
        from tests.unit.studies.test_study import _study, _value

        study = _study("find_pockets")
        study.start_step(0)
        study.finish_step(0, [_value(kind="pocket_candidates", produced_by=0)])
        assert study.thread(0) is None
        study.say(0, Turn(role="scientist", say="which one?", at=datetime.now(UTC)))
        study.say(0, Turn(role="agent", say="candidate 10", at=datetime.now(UTC)))
        thread = study.thread(0)
        assert thread is not None
        assert [t.role for t in thread.turns] == ["scientist", "agent"]
        assert len(study.conversations) == 1


class TestStudyProgress:
    """The study agent's claims are checked against the plan, not just phrased well."""

    def _study(self) -> Any:
        from tests.unit.studies.test_study import _study

        study = _study("fetch_pdb", "find_pockets")
        study.checks = ("Exactly one candidate is lined by His94 and His96.",)
        study.stop_conditions = ("The structure has no zinc.",)
        return study

    def test_a_check_cited_as_the_plan_wrote_it_survives(self) -> None:
        from daikonstudio.application.studies.study_chat import ProgressClaim, _check_progress

        points, dropped = _check_progress(
            [
                ProgressClaim(
                    about="check",
                    refers_to=[1],
                    label="not met",
                    cites=["Exactly one candidate is lined by His94 and His96."],
                )
            ],
            self._study(),
        )
        assert dropped == 0
        assert points[0].at == "check 1"
        assert points[0].evidence == "check"

    def test_a_criterion_the_plan_never_wrote_is_dropped(self) -> None:
        """The failure that matters: a plausible check, confidently attributed to
        a plan that does not contain it."""
        from daikonstudio.application.studies.study_chat import ProgressClaim, _check_progress

        points, dropped = _check_progress(
            [
                ProgressClaim(
                    about="check",
                    refers_to=[1],
                    label="not met",
                    cites=["Exactly one candidate is lined by His119."],
                )
            ],
            self._study(),
        )
        assert points == []
        assert dropped == 1

    def test_a_step_that_is_not_in_the_study_is_dropped(self) -> None:
        from daikonstudio.application.studies.study_chat import ProgressClaim, _check_progress

        points, dropped = _check_progress(
            [ProgressClaim(about="step", refers_to=[7], label="done", cites=["Find pockets"])],
            self._study(),
        )
        assert points == []
        assert dropped == 1

    def test_the_study_thread_is_kept_apart_from_the_stages(self) -> None:
        study = self._study()
        study.say(None, Turn(role="agent", say="where it stands", at=datetime.now(UTC)))
        study.say(0, Turn(role="agent", say="about one result", at=datetime.now(UTC)))
        assert study.thread(None) is not None
        assert study.thread(None) is not study.thread(0)
        assert len(study.conversations) == 2


class TestProposals:
    """What an agent may ask for is checked the way what it claims is checked.

    An offered change is a button the scientist will press, so a proposal that
    cannot run is worse than none: it reads as a way forward and is a dead end.
    """

    def _study(self) -> Any:
        from daikonstudio.domain.studies.study import SlotValue
        from tests.unit.studies.test_study import _study

        study = _study("fetch_pdb", "prepare_structure")
        study.values.append(
            SlotValue(
                id=uuid.uuid4(),
                key="structure",
                kind="structure",
                blob_key="b/1",
                summary={},
                produced_by=0,
                created_at=datetime.now(UTC),
            )
        )
        return study

    def _vet(self, *proposals: Any) -> Any:
        from daikonstudio.application.studies.proposals import vet
        from daikonstudio.infrastructure.tools.catalog import default_registry

        return vet(list(proposals), self._study(), default_registry())

    def test_a_rerun_with_settings_that_tool_takes_is_offered(self) -> None:
        from daikonstudio.application.planner.plan import Setting
        from daikonstudio.application.studies.proposals import Proposal

        kept, dropped = self._vet(
            Proposal(
                does="rerun",
                step=1,
                settings=[Setting(name="ph", value="6.5")],
                why="The site is acidic.",
            )
        )
        assert dropped == 0
        assert kept[0].settings == {"ph": "6.5"}
        assert kept[0].applied_at is None

    def test_a_setting_that_tool_does_not_take_costs_the_proposal(self) -> None:
        """A button that would fail the moment it is pressed."""
        from daikonstudio.application.planner.plan import Setting
        from daikonstudio.application.studies.proposals import Proposal

        kept, dropped = self._vet(
            Proposal(
                does="rerun", step=1, settings=[Setting(name="temperature", value="310")], why="x"
            )
        )
        assert kept == []
        assert dropped == 1

    def test_a_step_whose_input_nothing_has_produced_is_not_offered(self) -> None:
        from daikonstudio.application.studies.proposals import Proposal

        kept, dropped = self._vet(
            Proposal(does="add", tool_id="find_pockets", why="Find the cavities.")
        )
        # Pockets need a structure, which this study has, so this one stands.
        assert dropped == 0 and len(kept) == 1

        kept, dropped = self._vet(
            Proposal(does="add", tool_id="activity_cliffs", why="Look for cliffs.")
        )
        assert kept == [] and dropped == 1

    def test_an_automatic_step_is_never_the_agents_to_add(self) -> None:
        from daikonstudio.application.studies.proposals import Proposal

        kept, dropped = self._vet(
            Proposal(does="add", tool_id="prepare_structure", why="Prepare it again.")
        )
        assert kept == [] and dropped == 1

    def test_an_invented_tool_is_dropped(self) -> None:
        from daikonstudio.application.studies.proposals import Proposal

        kept, dropped = self._vet(Proposal(does="add", tool_id="dock_ligand", why="Dock it."))
        assert kept == [] and dropped == 1
