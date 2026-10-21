"""Turning a scientist's stated goal into a reviewable plan.

The planner is a compiler from intent to a plan, not a driver. It selects from
the tool catalogue and parameterizes nothing it cannot name; it does not execute,
and it does not decide. A plan that comes back incoherent is returned as a plan
with its problems attached rather than raised as an error, because the problems
are the most useful thing on the page: they say exactly where the model reached
for something that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.planner.ask_first import QUESTION_BUDGET
from daikonstudio.application.planner.catalogue import always_runs, catalogue, profile_text
from daikonstudio.application.planner.plan import (
    Answer,
    DraftPlan,
    PlannedStep,
    PlanProblem,
    expand,
    incoherences,
)
from daikonstudio.application.ports.llm import LanguageModels, ModelChoice
from daikonstudio.application.ports.table_profiler import TableProfile
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.domain.shared.errors import DomainError, ValidationError

# Written at the planner, not at the model: these are the house rules a plan has
# to satisfy, and they are the same whichever provider answers. Kept here rather
# than in a prompt file so that a change to them shows up in review beside the
# code that depends on them.
_SYSTEM_TEMPLATE = """\
You plan analyses for medicinal chemists inside Daikon Studio. You turn a stated \
goal into a plan a chemist will read and approve before anything runs.

Rules, in order of importance.

1. Use only tools from the catalogue below, by their exact id. If the goal needs \
something the catalogue cannot do, say so in a question rather than inventing a \
tool or pretending a different tool covers it.

2. Order steps so that every step's inputs are produced by an earlier step. A \
compound table is available from the start; everything else has to be made. A \
tool marked as needing a chemist produces options, and what the chemist approves \
becomes the value later steps use.

2a. Some entries are one step that runs several tools in order. Prefer one of \
those over listing its parts separately when you want the whole sequence, and \
reach for an individual part only when you genuinely want that part alone.

3. Ask about what is underspecified, and derive the questions from what you \
were actually given rather than from a checklist.

When a file's shape is shown below, read it first. Do not ask what you can see: \
the shape already lists, per column, how much of it is filled, what parses as \
what, the range of any numbers, and the values that do not fit the rest. Ask \
instead about what the shape cannot tell you -- what a column means, which of \
two plausible readouts is the one that matters, whether two batches were \
produced under the same conditions -- and quote the number that prompted the \
question, so a decision is being asked for rather than homework. If the shape \
shows something the scientist did not mention and would want to know about, \
raise it unasked.

Nothing in the shape is interpreted for you. A column of numbers is a column of \
numbers; whether it is a potency, a yield or a plate index is your inference and \
the scientist's to confirm.

Without a file, say in your questions which of them reading the data would have \
settled.

3a. Where the scientist has already answered, use the answer and do not ask \
again. Where they said they were not sure, or left a question unanswered, do not \
quietly pick for them: either add a step that settles it, or say plainly in the \
plan which assumption you made and what it would cost if it is wrong. An \
unanswered question resolved silently is the exact failure that asking first was \
meant to prevent.

3b. You may raise something that only became worth raising once you had their \
answers, but the same test applies and it is stricter here, because the plan is \
already written. Say what you will do and set `default` to it. A question that \
admits it "costs nothing and changes nothing", or that names the option usually \
worth having, has answered itself: take that answer, state it as an assumption, \
and move on. Leave `default` null only if the plan in front of you is genuinely \
wrong either way, which after a round of answers is rare.

3c. You have %(left)d questions left for this whole analysis, counting every \
round. Spend them on what would change the plan most. At zero, ask nothing: \
write the plan on stated assumptions, which you can always do.

4. State what would count as a result before anything runs, and state at least \
one condition under which the analysis should stop and report that the question \
cannot be answered this way. A plan that can only succeed is not a plan.

5. Gate a step when a person would want to look at its result before going on: \
after a judgement, after something that can quietly fail, and before anything \
expensive. Do not gate a step nobody would stop at.

6. Say which tools you considered and left out, and why. A short plan with stated \
omissions is worth more than a long one that uses everything.

7. %(always)s

Write for a working chemist. Be specific and brief. Do not flatter the goal, and \
do not promise a conclusion the analysis cannot support.
"""


def _answers(answers: tuple[Answer, ...]) -> str:
    """What the scientist said, including what they said they did not know.

    An unanswered question and a "not sure" are different from each other and
    both different from an answer, so they are written out rather than dropped.
    A question quietly treated as settled is the thing asking first was for.
    """
    lines = ["You asked, and the scientist replied:"]
    for item in answers:
        said = item.answer.strip() or "(no answer)"
        lines.append(f"- {item.ask}\n  -> {said}")
    return "\n".join(lines)


@dataclass(frozen=True, kw_only=True)
class DraftPlanCommand:
    goal: str
    # Which provider to ask. None takes the deployment's default.
    provider: str | None = None
    # The shape of the data, when the scientist supplied a file. Without it the
    # planner has to guess at what it could have read, and its questions become
    # homework rather than decisions.
    profile: TableProfile | None = None
    # What the scientist said to the questions asked first. An empty tuple is a
    # plan written on assumptions, which is allowed and is what the plan then
    # has to say about itself.
    answers: tuple[Answer, ...] = ()


@dataclass(frozen=True, kw_only=True)
class PlanDraft:
    """A plan with everything needed to judge it: who wrote it, and what is wrong
    with it. `problems` empty means it could be executed as written, not that it
    is a good plan -- that judgement is the scientist's."""

    goal: str
    plan: DraftPlan
    # The plan's steps with composites replaced by their parts: what will actually
    # run, in order, each naming the composite it came from.
    steps: list[PlannedStep]
    problems: list[PlanProblem]
    written_by: ModelChoice
    # How many questions the analysis may still ask. Shown to the scientist so
    # the loop is visibly bounded rather than feeling open-ended.
    questions_left: int


class WritePlan:
    def __init__(self, models: LanguageModels, tools: ToolRegistry) -> None:
        self._models = models
        self._tools = tools

    async def __call__(
        self, command: DraftPlanCommand, auth: AuthContext | None = None
    ) -> Result[PlanDraft, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None

        goal = command.goal.strip()
        if not goal:
            return Failure(ValidationError("Say what you want to find out."))

        model = self._models.get(command.provider)
        manifests = self._tools.manifests()
        parts = [f"Catalogue:\n{catalogue(manifests)}"]
        if command.profile is not None:
            parts.append(profile_text(command.profile))
        parts.append(f"The scientist's goal, in their words:\n{goal}")
        if command.answers:
            parts.append(_answers(command.answers))
        prompt = "\n\n".join(parts)
        remaining = max(QUESTION_BUDGET - len(command.answers), 0)
        system = _SYSTEM_TEMPLATE % {
            "always": always_runs(manifests),
            # Derived rather than tracked: the client sends back one entry per
            # question it was given, answered or not, so the count of replies
            # is the count of questions already spent.
            "left": remaining,
        }
        plan = await model.parse(system=system, prompt=prompt, schema=DraftPlan)
        # Trimmed to the budget here, not trusted to the prompt. The model has
        # ranked them, so the ones that survive are the ones it cared about.
        plan.questions = plan.questions[:remaining]
        steps = expand(plan, self._tools)

        return Success(
            PlanDraft(
                goal=goal,
                plan=plan,
                steps=steps,
                problems=incoherences(steps, self._tools),
                written_by=model.choice,
                questions_left=remaining,
            )
        )
