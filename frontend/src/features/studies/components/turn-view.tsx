"use client";

import { useTools } from "@/features/tools";
import { Button } from "@/shared/components/ui/button";
import { cn } from "@/shared/lib/utils";
import { useApplyProposal, useStudy } from "../hooks/use-studies";
import type { Study, StudyProposal, StudyStep, Turn } from "../types";
import { RUN_UNDER_WAY } from "../types";

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
      return `Take ${step} out of the plan`;
    case "swap":
      return `Use ${tool} for ${step}${listed}`;
    case "settings":
      return `Change the settings of ${step}${listed}`;
    case "gate":
      return action.gated ? `Stop after ${step} for you to look` : `Let ${step} run on`;
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
                ? "Done. The step is waiting for you to run it."
                : "Done."}
            </p>
          ) : (
            <div className="flex flex-wrap items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                disabled={apply.isPending || running}
                onClick={() => apply.mutate(proposal.id)}
              >
                Do it
              </Button>
              {running && <p className="text-xs text-muted-foreground">{RUN_UNDER_WAY}</p>}
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
  return quoting ? `Did: ${done}, on your words: '${quoting}'` : `Did: ${done}`;
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
        {failed ? `Could not look at ${where}: ${turn.say}` : turn.say}
      </p>
    );
  }
  const agent = turn.role === "agent";
  const dropped =
    turn.dropped > 0 &&
    `${turn.dropped} ${turn.dropped === 1 ? "claim" : "claims"} dropped, unsupported by this result`;
  return (
    <div
      className={cn(
        "min-w-0 space-y-2 rounded-sm border p-3 text-sm",
        agent ? "bg-muted/40" : "border-dashed",
      )}
    >
      {turn.about && (
        <p className="text-xs text-muted-foreground">About {aboutStep(steps, turn.about)}</p>
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
      {turn.points.length > 0 && (
        <ul className="space-y-1">
          {turn.points.map((point) => (
            <li key={`${point.step}-${point.at}-${point.label}`}>
              {[
                numbered(steps, point.step),
                point.at,
                point.label,
                point.evidence.replace(/_/g, " "),
                point.cites.join(", "),
              ]
                .filter(Boolean)
                .join(" · ")}
            </li>
          ))}
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
      {turn.unresolved.length > 0 && (
        <ul className="space-y-0.5 text-muted-foreground">
          {turn.unresolved.map((line) => (
            <li key={line}>Unresolved: {line}</li>
          ))}
        </ul>
      )}
      {turn.looked.length > 0 && (
        <p className="text-xs text-muted-foreground">Looked at {turn.looked.join("; ")}.</p>
      )}
      {agent && (turn.written_by || dropped) && (
        <p className="text-xs text-muted-foreground">
          {[turn.written_by, dropped].filter(Boolean).join(" · ")}
        </p>
      )}
    </div>
  );
}
