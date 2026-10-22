"use client";

import { Button } from "@/shared/components/ui/button";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { cn } from "@/shared/lib/utils";
import { Loader2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useApplyProposal, useSay, useStageThread } from "../hooks/use-studies";
import type { StageTurn, StudyProposal, Thread } from "../types";

/**
 * One claim the agent made, after the check.
 *
 * Shown as what it points at rather than as a sentence, because that is what it
 * is: a rank, a name, and the residues the result itself carries. A reader can
 * take any of it back to the table above and find it.
 */
function Points({ turn }: { turn: StageTurn }) {
  if (turn.points.length === 0) return null;
  return (
    <ul className="space-y-1">
      {turn.points.map((point) => (
        <li key={`${point.at}-${point.label}`} className="text-sm">
          <span className="font-medium">{point.at}</span>
          {" · "}
          {point.label}
          <span className="text-muted-foreground">
            {" · "}
            {point.evidence.replace(/_/g, " ")}
            {point.cites.length > 0 && ` · ${point.cites.join(", ")}`}
          </span>
        </li>
      ))}
    </ul>
  );
}

/**
 * A change the agent would like made, and the button that makes it.
 *
 * Confirmed, never automatic. The plan is worth something because it was written
 * before the data; an agent that could edit it after seeing a result would turn
 * it into a description of what happened. So the agent names the change exactly
 * and the scientist decides.
 */
function Proposals({ studyId, proposals }: { studyId: string; proposals: StudyProposal[] }) {
  const apply = useApplyProposal(studyId);
  if (proposals.length === 0) return null;
  return (
    <ul className="space-y-2">
      {proposals.map((proposal) => {
        const settings = Object.entries(proposal.settings ?? {});
        const what =
          proposal.does === "rerun"
            ? `Run step ${proposal.step} again`
            : `Add ${proposal.tool_id} to the plan`;
        return (
          <li key={proposal.id} className="space-y-1 rounded-sm border border-dashed p-2">
            <p className="text-sm">
              <span className="font-medium">{what}</span>
              {settings.length > 0 && (
                <span className="text-muted-foreground">
                  {" with "}
                  {settings.map(([name, value]) => `${name} ${String(value)}`).join(", ")}
                </span>
              )}
            </p>
            <p className="text-sm text-muted-foreground">{proposal.why}</p>
            {proposal.applied_at ? (
              <p className="text-xs text-muted-foreground">
                Plan changed. The step is waiting for you to run it.
              </p>
            ) : (
              <Button
                size="sm"
                disabled={apply.isPending}
                onClick={() => apply.mutate(proposal.id)}
              >
                Modify plan
              </Button>
            )}
          </li>
        );
      })}
      {apply.isError && (
        <li className="text-sm text-destructive" role="alert">
          {apply.error?.message}
        </li>
      )}
    </ul>
  );
}

function TurnView({ studyId, turn }: { studyId: string; turn: StageTurn }) {
  const agent = turn.role === "agent";
  return (
    <div
      className={cn(
        "space-y-2 rounded-sm border p-3",
        agent ? "bg-muted/40" : "border-dashed bg-background",
      )}
    >
      <p className="whitespace-pre-wrap text-sm">{turn.say}</p>
      <Points turn={turn} />
      <Proposals studyId={studyId} proposals={turn.proposals ?? []} />
      {turn.unresolved.length > 0 && (
        <ul className="space-y-0.5 text-sm text-muted-foreground">
          {turn.unresolved.map((line) => (
            <li key={line}>Unresolved: {line}</li>
          ))}
        </ul>
      )}
      {turn.proposes.length > 0 && (
        <p className="text-sm text-muted-foreground">
          Worth running next: {turn.proposes.join(", ")}
        </p>
      )}
      {agent && (
        <p className="text-xs text-muted-foreground">
          {turn.written_by}
          {/* Shown, not swallowed: a turn that had claims taken off it is worth
              less than one that did not, and only the reader can weigh that. */}
          {turn.dropped > 0 &&
            ` · ${turn.dropped} ${turn.dropped === 1 ? "claim" : "claims"} dropped, unsupported by this result`}
        </p>
      )}
    </div>
  );
}

/**
 * The conversation at one stage.
 *
 * It opens only when asked. The result is what the scientist came to read, and
 * an agent that speaks first would be read first — which is the failure this
 * whole design is arranged against, since confident prose outruns a table.
 */
export function StageChat({
  studyId,
  position,
  title = "Talk about this result",
  focusToken = 0,
}: {
  studyId: string;
  /** Null is the study's own conversation rather than one stage's. */
  position: number | null;
  title?: string;
  /** Bumped by the caller to put the cursor here. Any change counts, so asking
   *  for the conversation already open still lands somewhere. */
  focusToken?: number;
}) {
  const thread = useStageThread(studyId, position);
  const say = useSay(studyId, position);
  const [message, setMessage] = useState("");
  const box = useRef<HTMLTextAreaElement>(null);

  const data: Thread | null | undefined = thread.data;

  // `thread.isLoading` is a dependency because the textarea does not exist while
  // the conversation is still being fetched: asking for a stage that has not
  // loaded would otherwise focus nothing and look like a dead button.
  useEffect(() => {
    // Nothing to focus while the conversation is still arriving; the effect
    // runs again when it has, which is what makes a press on a stage that is
    // not yet loaded still land.
    if (focusToken === 0 || thread.isLoading) return;
    box.current?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
    box.current?.focus({ preventScroll: true });
  }, [focusToken, thread.isLoading]);
  const busy = say.isPending;

  function send(text: string | null) {
    say.mutate(text, { onSuccess: () => setMessage("") });
  }

  if (thread.isLoading) return <Skeleton className="h-10 w-full rounded-sm" />;

  return (
    <section className="space-y-2">
      <h4 className="text-xs uppercase tracking-wide text-muted-foreground">{title}</h4>
      {data?.turns.map((turn, i) => (
        <TurnView key={`${turn.at}-${i}`} studyId={studyId} turn={turn} />
      ))}
      {!data && !busy && (
        <p className="text-sm text-muted-foreground">
          Nothing said yet. The agent reads the same result you can see, and anything it claims is
          checked against it first.
        </p>
      )}
      {data?.handoff && (
        /* Shown although it is addressed to another agent, not to the scientist:
           it is what a study-wide agent will read instead of this whole
           exchange, and a summary nobody can see is one nobody can correct. */
        <p className="rounded-sm border border-dashed p-2 text-sm text-muted-foreground">
          <span className="text-foreground">What leaves this stage </span>
          {data.handoff}
        </p>
      )}
      {busy && (
        <p className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" aria-hidden="true" />
          Reading the result
        </p>
      )}
      <div className="flex items-start gap-2">
        <label className="sr-only" htmlFor={`say-${position ?? "study"}`}>
          Ask about this result
        </label>
        <textarea
          ref={box}
          id={`say-${position ?? "study"}`}
          rows={2}
          value={message}
          disabled={busy}
          onChange={(e) => setMessage(e.target.value)}
          placeholder={data ? "Ask about this result" : "Ask about this result, or just open it"}
          className="min-w-0 flex-1 rounded-sm border bg-background p-2 text-sm"
        />
        <Button
          size="sm"
          disabled={busy}
          onClick={() => send(message.trim() ? message.trim() : null)}
        >
          {data ? "Send" : "Open"}
        </Button>
      </div>
      {say.isError && (
        <p className="text-sm text-destructive" role="alert">
          The agent could not answer. {say.error?.message}
        </p>
      )}
    </section>
  );
}

/**
 * The conversation, docked beside the results rather than under them.
 *
 * A claim is only checkable if the table it points at is still on screen. Under
 * a step card it was not: a reader scrolled past the candidates and the viewer
 * to reach the agent, and then had to scroll back to test a single residue. One
 * panel, pinned, with the stages as its tabs.
 *
 * The study-wide agent is not built. When it is, it joins this list as another
 * tab rather than becoming a second place to look.
 */
export function StageChatRail({
  studyId,
  stages,
  active,
  onSelect,
  focusToken = 0,
}: {
  studyId: string;
  stages: { position: number | null; name: string }[];
  active: number | null;
  onSelect: (position: number | null) => void;
  focusToken?: number;
}) {
  if (stages.length === 0) return null;
  const current = stages.find((s) => s.position === active) ?? stages[0];
  return (
    <div className="space-y-3 rounded-sm border p-3">
      {stages.length > 1 && (
        <div className="flex flex-wrap gap-1">
          {stages.map((stage) => (
            <Button
              key={stage.position}
              size="sm"
              variant={stage.position === current.position ? "secondary" : "ghost"}
              onClick={() => onSelect(stage.position)}
            >
              {stage.name}
            </Button>
          ))}
        </div>
      )}
      <StageChat
        studyId={studyId}
        position={current.position}
        title={current.name}
        focusToken={focusToken}
      />
    </div>
  );
}
