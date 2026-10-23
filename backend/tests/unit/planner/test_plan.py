"""What a drafted plan must never get away with.

A model writes plans that read well. The two failures these cover are the ones
that read perfectly and cannot execute: a tool that does not exist, and a step
reaching for something nothing upstream has made. Both are caught mechanically,
which is the entire reason tools declare what they consume and produce.
"""

from __future__ import annotations

from daikonstudio.application.planner.catalogue import brought, starting_note
from daikonstudio.application.planner.plan import DraftPlan, Setting, Step, expand, incoherences
from daikonstudio.application.tools.manifest import SlotKind
from daikonstudio.infrastructure.tools.catalog import default_registry


def _plan(*tool_ids: str, gated: bool = False) -> DraftPlan:
    return DraftPlan(
        restated_goal="g",
        steps=[Step(tool_id=t, purpose="p", gated=gated) for t in tool_ids],
    )


def _chosen(steps: list[object]) -> list[str]:
    """Only what the planner picked. The always-run prologue is prepended to
    every plan, so an assertion about the planner's choices has to exclude it."""
    return [s.tool_id for s in steps if not s.always]  # type: ignore[attr-defined]


def test_a_coherent_sar_plan_has_no_problems() -> None:
    registry = default_registry()
    plan = _plan("standardize", "find_cores", "build_sar_table", "analyse_pairs")
    assert incoherences(expand(plan, registry), registry) == []


def test_an_invented_tool_is_caught() -> None:
    registry = default_registry()
    problems = incoherences(expand(_plan("standardize", "magic_sar"), registry), registry)
    assert len(problems) == 1
    assert "magic_sar" in problems[0].message
    assert problems[0].step == 2


def test_a_step_reaching_for_something_nothing_made_is_caught() -> None:
    """Decomposing needs a core, and a core only exists once a chemist has
    approved one. A plan that skips straight to the SAR table reads fine."""
    registry = default_registry()
    problems = incoherences(expand(_plan("standardize", "build_sar_table"), registry), registry)
    assert len(problems) == 1
    assert "core" in problems[0].message


def test_an_empty_plan_is_a_problem() -> None:
    registry = default_registry()
    assert incoherences(expand(_plan(), registry), registry)[0].message == "The plan has no steps."


def test_approving_a_candidate_satisfies_a_later_step() -> None:
    """`find_cores` outputs candidates, not a core. The plan is coherent only
    because approving one resolves to a core, which is the seam the whole
    provenance story rests on."""
    registry = default_registry()
    assert incoherences(expand(_plan("find_cores", "decompose_rgroups"), registry), registry) == []


class TestComposites:
    def test_a_composite_expands_into_its_parts(self) -> None:
        registry = default_registry()
        steps = expand(_plan("build_sar_table"), registry)
        assert _chosen(steps) == ["decompose_rgroups", "name_substituents"]
        assert all(s.expanded_from == "build_sar_table" for s in steps if not s.always)

    def test_a_step_that_asks_a_chemist_gates_even_inside_a_composite(self) -> None:
        """The planner only decides whether to stop at the end of a composite.
        A step that needs an approval gates because of what it is, or a plan
        could quietly skip the pause on a decision."""
        registry = default_registry()
        steps = expand(_plan("analyse_pairs", gated=False), registry)
        gated = {s.tool_id for s in steps if s.gated}
        assert gated == {"activity_cliffs"}

    def test_gating_a_composite_gates_its_last_step(self) -> None:
        registry = default_registry()
        steps = expand(_plan("build_sar_table", gated=True), registry)
        assert [s.gated for s in steps if not s.always] == [False, True]

    def test_composites_do_not_nest(self) -> None:
        """One level, by construction. Composites of composites would be a second
        plan language, and `expand` would need to recurse and could loop."""
        registry = default_registry()
        composites = {m.id for m in registry.manifests() if m.expands_to}
        for manifest in registry.manifests():
            assert not (set(manifest.expands_to) & composites), f"{manifest.id} nests a composite"

    def test_every_composite_is_internally_coherent(self) -> None:
        """A composite's own parts have to run in the order it lists them, given
        only what the composite says it takes. Otherwise the planner picks one
        choice and gets an impossible sequence."""
        registry = default_registry()
        for manifest in registry.manifests():
            if not manifest.expands_to:
                continue
            plan = DraftPlan(
                restated_goal="g",
                steps=[
                    Step(
                        tool_id=manifest.id,
                        purpose="p",
                        gated=False,
                        settings=[Setting(name="activity_column", value="pIC50")],
                    )
                ],
            )
            steps = expand(plan, registry)
            problems = incoherences(
                steps,
                registry,
                # The prologue is prepended to every expansion and legitimately
                # needs the compound table the composite was handed.
                starting_with={slot.kind for slot in manifest.inputs} | {SlotKind.COMPOUND_TABLE},
            )
            assert problems == [], f"{manifest.id}: {[p.message for p in problems]}"


def test_preparation_alone_is_still_an_empty_plan() -> None:
    """The prologue runs whatever the planner said, so "no steps" has to mean no
    steps anyone chose. A plan that only prepares the data answered nothing."""
    registry = default_registry()
    steps = expand(_plan(), registry)
    assert [s.tool_id for s in steps] == ["standardize"]
    assert incoherences(steps, registry)[0].message == "The plan has no steps."


class TestWhatThePlannerIsTold:
    """The prompt may not restate anything the catalogue already knows.

    Prose repeating today's catalogue becomes a lie the moment the catalogue
    changes, and the model goes on asserting it confidently. Anything the
    planner is told about the tools has to be generated from the tools.
    """

    def test_the_always_runs_note_names_what_the_catalogue_marks(self) -> None:
        from daikonstudio.application.planner.catalogue import always_runs

        registry = default_registry()
        note = always_runs(registry.manifests())
        for manifest in registry.manifests():
            if manifest.automatic:
                assert manifest.name in note
            else:
                assert manifest.name not in note

    def test_a_catalogue_with_nothing_automatic_says_so(self) -> None:
        from daikonstudio.application.planner.catalogue import always_runs
        from daikonstudio.application.tools.registry import ToolRegistry

        plain = ToolRegistry(tuple(m for m in default_registry().manifests() if not m.automatic))
        assert "every step in your plan is one you chose" in always_runs(plain.manifests())

    def test_the_catalogue_says_what_the_planner_can_act_on(self) -> None:
        """Added-for-you entries are marked so the model does not choose them,
        but one that takes parameters must still be listable to set them, or an
        answer about pH or which ligand to strip can never reach the tool;
        external tools say so because the gating rule asks for a pause after
        anything that can quietly fail; and the slot vocabulary is spelled one
        way, because the note on what was brought already uses the spaced form."""
        from daikonstudio.application.planner.catalogue import catalogue

        registry = default_registry()
        text = catalogue(registry.manifests())
        for manifest in registry.manifests():
            entry = next(b for b in ("\n" + text).split("\n- ") if b.startswith(f"{manifest.id}:"))
            added = "added to every plan for you" in entry
            assert added is manifest.automatic, manifest.id
            if manifest.automatic:
                expected = (
                    "list it only to set its parameters" if manifest.params else "never list it"
                )
                assert expected in entry, manifest.id
            assert ("runs outside Studio" in entry) is manifest.external, manifest.id
        assert "compound_table" not in text
        assert "lane" not in text


class TestAskingSparingly:
    """A question the planner could have answered itself is not a question.

    Without this the loop never ends: every answer gives the model enough
    context to see one more choice, and the scientist is three rounds into an
    interview instead of reading a plan. The test is not "does it change the
    analysis" -- the questions that caused this all passed that -- it is
    "could you have defended a default".
    """

    def test_a_question_with_a_default_does_not_block(self) -> None:
        from daikonstudio.application.planner.plan import Question

        refinement = Question(
            ask="Call out the two compounds with no cell activity?",
            why="Costs nothing and changes nothing in the analysis.",
            default="Yes, call them out.",
        )
        assert refinement.blocking is False

    def test_a_question_without_a_default_blocks(self) -> None:
        from daikonstudio.application.planner.plan import Question

        blocking = Question(ask="Which column is the readout?", why="Everything rests on it.")
        assert blocking.blocking is True

    def test_the_budget_counts_every_question_already_put(self) -> None:
        """Unanswered still counts as asked. The client sends one entry per
        question it was shown, so the spend is the length of that list and
        needs no separate counter to drift out of step."""
        from daikonstudio.application.planner.plan import Answer
        from daikonstudio.application.studies.prompts import QUESTION_BUDGET

        answers = tuple(Answer(ask=f"q{i}", answer="") for i in range(4))
        assert max(QUESTION_BUDGET - len(answers), 0) == 6
        spent = tuple(Answer(ask=f"q{i}", answer="") for i in range(QUESTION_BUDGET + 3))
        assert max(QUESTION_BUDGET - len(spent), 0) == 0


class TestWhatTheScientistBrought:
    """An automatic step applies only when what it needs is already in hand.

    Preparing a structure always happens before anything touches one, but a plan
    over a compound table alone has no structure to prepare. Prepending it
    regardless would put a step in the plan for a file nobody supplied -- which
    is how a second automatic tool turns the first one's convenience into a bug.
    """

    def test_structure_preparation_is_skipped_without_a_structure(self) -> None:
        registry = default_registry()
        steps = expand(_plan("find_cores"), registry)
        assert [s.tool_id for s in steps] == ["standardize", "find_cores"]

    def test_structure_preparation_is_added_when_one_was_supplied(self) -> None:
        registry = default_registry()
        steps = expand(
            _plan("find_pockets"),
            registry,
            starting_with={SlotKind.COMPOUND_TABLE, SlotKind.STRUCTURE},
        )
        assert [s.tool_id for s in steps] == ["standardize", "prepare_structure", "find_pockets"]
        assert [s.always for s in steps] == [True, True, False]

    def test_the_planner_is_never_told_about_data_it_was_not_given(self) -> None:
        """Claiming a slot nobody supplied is how the planner came to ask what
        the compound table was and what should happen to it -- unanswerable,
        because there was no file, and it reads as invention when in fact it
        was told. Checked against every slot rather than that one, since the
        next wrong claim will be about a different kind."""
        note = starting_note(brought(None))
        assert [k.value for k in SlotKind if k.value.replace("_", " ") in note] == []

    def test_an_automatic_step_follows_the_plan_not_the_upload(self) -> None:
        """An automatic tool prepares a slot, and belongs in a plan exactly when
        some step expects to be handed that slot -- whether or not a file has
        arrived, because the upload is optional and a chemist can describe a
        series before attaching it.

        The pairs come from the registry rather than being named here. Pinning
        this to two tools would keep passing while the rule was wrong for a
        third, which is the failure that put a compound-cleaning step into a
        plan about a protein.
        """
        registry = default_registry()
        optional = [m for m in registry.manifests() if not m.automatic]
        automatic = [m for m in registry.manifests() if m.automatic]
        assert automatic, "nothing runs automatically, so there is nothing to place"

        for auto in automatic:
            prepares = {slot.kind for slot in auto.inputs}
            wants = [m for m in optional if prepares <= {s.kind for s in m.inputs}]
            ignores = [m for m in optional if not prepares & {s.kind for s in m.inputs}]
            assert wants and ignores, f"no pair in the catalogue can place {auto.id}"

            for step in (wants[0], ignores[0]):
                placed = expand(_plan(step.id), registry, starting_with=set())
                expected = step in wants
                assert (auto.id in [s.tool_id for s in placed]) is expected, (
                    f"{auto.id} with a plan that runs {step.id}"
                )

    def test_a_structure_fetched_mid_plan_is_prepared_before_it_is_used(self) -> None:
        """An automatic step goes in when its input exists, not only at the
        start. A structure fetched by id in step one needs preparing as much as
        one attached up front; prepending-or-nothing silently skipped it."""
        registry = default_registry()
        plan = _plan("fetch_pdb", "find_pockets")
        plan.steps[0].settings = [Setting(name="pdb_id", value="4TRO")]
        steps = expand(plan, registry, starting_with=set())
        assert [s.tool_id for s in steps] == ["fetch_pdb", "prepare_structure", "find_pockets"]
        assert [s.always for s in steps] == [False, True, False]
        assert incoherences(steps, registry) == []

    def test_a_goal_naming_an_id_can_be_planned_without_a_file(self) -> None:
        """The note the planner reads must not deny what the catalogue can now
        do: a step that takes nothing makes its output from the goal."""
        note = starting_note(brought(None))
        assert "takes nothing" in note
        assert "nothing in the catalogue produces" not in note

    def test_the_always_run_note_does_not_promise_a_prologue(self) -> None:
        """An automatic step now goes in wherever its input first appears, so a
        note saying it has "already happened before your first step" would be a
        lie for a structure fetched in step one."""
        from daikonstudio.application.planner.catalogue import always_runs

        note = always_runs(default_registry().manifests())
        assert "already happened before your first step" not in note
        assert "first produces what it needs" in note

    def test_a_pocket_is_approved_rather_than_computed(self) -> None:
        """The same seam as a core, for the same reason: the top-ranked pocket
        is often a crystallisation artefact, and which site is worth aiming at
        is a question about the biology rather than the geometry."""
        registry = default_registry()
        assert not registry.producing(SlotKind.POCKET)
        approvers = {m.id for m in registry.manifests() if m.resolves_to is SlotKind.POCKET}
        assert approvers == {"find_pockets", "predict_pockets"}
