"use client";

import { useTools } from "@/features/tools";
import { Button } from "@/shared/components/ui/button";
import { cn } from "@/shared/lib/utils";
import { useApplyProposal, useStudy } from "../hooks/use-studies";
import type { Study, StudyProposal, StudyStep, Turn } from "../types";
import { STEPS_REASON_ID, stepAnchor } from "../types";
import { Fold } from "./fold";

/**
 * A step as the thread names it, numbered from 1 as the agent counts. A turn keeps the ids of
 * the steps it was about, and an edit can take one out after it was written, so a missing id
 * reads as gone rather than breaking the turn.
 */
function numbered(steps: StudyStep[], id: unknown): string | null {
  if (typeof id !== "string") return null;
  const step = steps.find((s) => s.id === id);
  return step ? `step ${step.position + 1}` : "a removed step";
}

/** A step with its name, for what a message or a turn is about. */
export function aboutStep(steps: StudyStep[], id: string): string {
  const step = steps.find((s) => s.id === id);
  return step ? `step ${step.position + 1} (${step.name})` : "a removed step";
}

/** What a proposal would do, in words, from the change its button keeps. */
function proposed(
  action: Record<string, unknown>,
  steps: StudyStep[],
  toolName: (id: string) => string,
): string {
  const step = numbered(steps, action.step_id) ?? "a step";
  const tool = typeof action.tool_id === "string" ? toolName(action.tool_id) : "a tool";
  const given =
    typeof action.settings === "object" && action.settings !== null ? action.settings : {};
  const settings = Object.entries(given).map(([name, value]) => `${name} ${String(value)}`);
  const listed = settings.length > 0 ? `: ${settings.join(", ")}` : "";
  switch (action.does) {
    case "insert": {
      const before = numbered(steps, action.before_id);
      return `Add ${tool} ${before ? `before ${before}` : "at the end"}${listed}`;
    }
    case "remove":
      return `Remove ${step} from the plan`;
    case "swap":
      return `Use ${tool} for ${step}${listed}`;
    case "settings":
      return `Change the settings of ${step}${listed}`;
    case "gate":
      return action.gated
        ? `Pause after ${step} for review`
        : `Continue after ${step} without a review pause`;
    case "approve":
      return `Use candidate ${String(action.candidate)} of ${step}`;
    case "run": {
      const start = numbered(steps, action.step_id);
      if (start && action.step_id === action.until_id) return `Run ${start}`;
      const until = numbered(steps, action.until_id);
      return `Run from ${start ?? "the next step waiting"} to ${until ?? "the next stop"}`;
    }
    default:
      return String(action.does);
  }
}

/**
 * Changes the agent would like made, each with the button that makes it. Confirmed, never
 * automatic: the press is the scientist's word, and the agent only names the change exactly.
 */
function Proposals({
  studyId,
  proposals,
  study,
}: {
  studyId: string;
  proposals: StudyProposal[];
  study: Study | undefined;
}) {
  const apply = useApplyProposal(studyId);
  const tools = useTools();
  const toolName = (id: string) =>
    tools.data?.find((tool) => tool.id === id)?.name ?? id.replace(/_/g, " ");
  const running = study?.running ?? false;
  return (
    <ul className="space-y-2">
      {proposals.map((proposal) => (
        <li
          key={proposal.id}
          className="space-y-1 rounded-sm border border-dashed bg-background p-2"
        >
          <p className="font-medium">{proposed(proposal.action, study?.steps ?? [], toolName)}</p>
          {proposal.why && <p className="text-muted-foreground">{proposal.why}</p>}
          {proposal.applied_at ? (
            <p className="text-xs text-muted-foreground">
              {["insert", "swap", "settings"].includes(String(proposal.action.does))
                ? "Proposal applied. The step is ready for execution."
                : "Proposal applied."}
            </p>
          ) : (
            <div className="flex flex-wrap items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                disabled={apply.isPending || running}
                aria-describedby={running ? STEPS_REASON_ID : undefined}
                onClick={() => apply.mutate(proposal.id)}
              >
                Apply proposal
              </Button>
            </div>
          )}
        </li>
      ))}
      {apply.error && (
        <li className="text-destructive" role="alert">
          {apply.error.message}
        </li>
      )}
    </ul>
  );
}

/** What the agent did on the scientist's word, quoting the words it acted on. */
function did({ detail, quoting }: { detail: string; quoting: string }): string {
  const done = detail.replace(/\.$/, "");
  return quoting
    ? `Action completed: ${done}. User instruction: “${quoting}”.`
    : `Action completed: ${done}.`;
}

/**
 * One turn of the thread. The scientist's words in a dashed box; the agent's on a muted ground
 * with everything that survived the check against the result; an event in one grey line. A
 * silent observation is a tick on its step's card, not a line here.
 */
export function TurnView({
  studyId,
  turn,
  answering = false,
}: {
  studyId: string;
  turn: Turn;
  /** The turn's questions are still open and answered below it on cards, not listed here. */
  answering?: boolean;
}) {
  const { data: study } = useStudy(studyId);
  const steps = study?.steps ?? [];
  if (turn.silent) return null;
  if (turn.role === "event") {
    const failed = turn.kind === "observe_failed";
    const where = turn.about ? aboutStep(steps, turn.about) : "a step";
    return (
      <p className="px-1 text-xs text-muted-foreground">
        {failed ? `Unable to review ${where}: ${turn.say}` : turn.say}
      </p>
    );
  }
  const agent = turn.role === "agent";
  // Said by what the turn could drop: an observation only claims, while a reply or a planning
  // turn also edits and proposes, and a refused instruction must not read as a weak claim.
  const dropped =
    turn.dropped > 0 &&
    (turn.kind === "observe"
      ? // Two reasons now, and saying only the first would misreport the second: a claim
        // about a step that has not run is dropped for having nothing in it, not for
        // being unsupported.
        `${turn.dropped} ${turn.dropped === 1 ? "claim" : "claims"} omitted because supporting results were unavailable or insufficient.`
      : `${turn.dropped} ${turn.dropped === 1 ? "item" : "items"} excluded: unsupported claims or changes incompatible with the current plan.`);
  // A point about a check or a stop condition is the standing state of the plan, and the
  // panel under the steps now holds that, with the criterion's own words beside it. Repeating
  // it here put the whole criterion in the thread on every turn, including the ones that said
  // only that a condition had not fired. What reads on the result itself stays.
  const onResult = turn.points
    .filter((point) => point.evidence !== "check" && point.evidence !== "stop_condition")
    // What the step did first, then what was read in it. Left alone, the line summarising
    // the step landed wherever the model happened to put it, often last and after a
    // comparison with an earlier step, so the list read as though it changed subject twice.
    .sort((a, b) => Number(b.evidence === "step") - Number(a.evidence === "step"));
  /* How this turn's own step is written, so a point that only repeats it can say less. */
  const ownStep = turn.about ? numbered(steps, turn.about) : null;
  return (
    <div
      className={cn(
        "min-w-0 space-y-2 rounded-sm border p-3 text-sm",
        agent ? "bg-muted/40" : "border-dashed",
      )}
    >
      {turn.about && (
        <p className="text-xs text-muted-foreground">
          About{" "}
          {/* The thread scrolls in its own column beside the steps, so "step 2" is a place
              the reader otherwise has to find by hand. */}
          <button
            type="button"
            className="underline underline-offset-2 hover:text-foreground"
            onClick={() =>
              document
                .getElementById(stepAnchor(String(turn.about)))
                ?.scrollIntoView({ behavior: "smooth", block: "start" })
            }
          >
            {aboutStep(steps, turn.about)}
          </button>
        </p>
      )}
      {turn.say && <p className="whitespace-pre-wrap">{turn.say}</p>}
      {/* The thread does not say whether a planning turn carried a plan or edits, so any
          planning turn on a study that has steps says the plan moved. */}
      {agent && turn.kind === "planning" && steps.length > 0 && (
        <p className="text-xs text-muted-foreground">Plan updated.</p>
      )}
      {!answering && turn.questions.length > 0 && (
        <ul className="space-y-1">
          {turn.questions.map((question) => {
            const given = study?.answers.findLast((a) => a.ask === question.ask)?.answer;
            return (
              <li key={question.ask}>
                <p>{question.ask}</p>
                {given && <p className="text-muted-foreground">You answered: {given}</p>}
              </li>
            );
          })}
        </ul>
      )}
      {onResult.length > 0 && (
        /* Ruled off from the prose: above is the argument, here is what it rests on. */
        <ul className="space-y-1.5 border-t pt-2">
          {onResult.map((point) => {
            // Name the step only when it is not the one this turn is about. Half of these
            // lines used to open with the same "step 5", which pushed the one part worth
            // reading, the label, into the middle of five identical dot-separated fields.
            const elsewhere = point.step && point.step !== turn.about;
            // Not when it only repeats the heading this turn already carries. A point about
            // the step in hand opened "step 5" above a label that began "5 learned patches",
            // which read as a stutter and said nothing the heading had not.
            const here = point.at === ownStep;
            const where = [elsewhere && numbered(steps, point.step), !here && point.at]
              .filter(Boolean)
              .join(" ");
            // Residues under a claim about residues need no label; a ligand overlap or an
            // annotated site does, because then the cites are not what you would assume.
            // A claim about the step cites the step's name, which the heading already gives.
            const kind =
              point.evidence === "lining_residues" ? "" : point.evidence.replace(/_/g, " ");
            const cites = point.evidence === "step" ? "" : point.cites.join(", ");
            return (
              <li key={`${point.step}-${point.at}-${point.label}`}>
                <p>
                  {where && <span className="text-muted-foreground">{where} </span>}
                  {point.label}
                </p>
                {cites && (
                  /* Reference, not argument: the residue list is the longest thing on the
                     line and the least of what the reader is being told. */
                  <p className="text-xs text-muted-foreground">
                    {[kind, cites].filter(Boolean).join(": ")}
                  </p>
                )}
              </li>
            );
          })}
        </ul>
      )}
      {turn.proposals.length > 0 && (
        <Proposals studyId={studyId} proposals={turn.proposals} study={study} />
      )}
      {turn.actions.length > 0 && (
        <ul className="space-y-0.5">
          {turn.actions.map((action) => (
            <li key={`${action.does}-${action.detail}`}>{did(action)}</li>
          ))}
        </ul>
      )}
      {/* Folded: what a turn could not settle is worth keeping and not worth re-reading on
          every scroll past it. The count is on the line, so nothing is hidden silently. */}
      {turn.unresolved.length > 0 && (
        <Fold label={`${turn.unresolved.length} unresolved`}>
          <ul className="space-y-0.5">
            {turn.unresolved.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </Fold>
      )}
      {turn.looked.length > 0 && (
        <p className="text-xs text-muted-foreground">Data reviewed: {turn.looked.join("; ")}.</p>
      )}
      {agent && (turn.written_by || dropped) && (
        <p className="text-xs text-muted-foreground">
          {[turn.written_by, dropped].filter(Boolean).join(" · ")}
        </p>
      )}
    </div>
  );
}
