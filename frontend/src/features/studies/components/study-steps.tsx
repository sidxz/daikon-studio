"use client";

import { useApprove, useRunStep, useStudy, useSwapTool, useThread } from "../hooks/use-studies";
import { RUN_UNDER_WAY, type StudyStep, type Turn } from "../types";
import { StepCard } from "./step-card";

/**
 * Whether the agent's newest look at a step's current result found nothing to add. Only an
 * observation made after the step last started is about that result, and a failure to observe
 * after it means nobody looked.
 */
function quiet(step: StudyStep, turns: Turn[]): boolean {
  if ((step.status !== "done" && step.status !== "failed") || !step.started_at) return false;
  const since = Date.parse(step.started_at);
  const last = turns.findLast(
    (t) =>
      t.about === step.id &&
      (t.kind === "observe_failed" || (t.kind === "observe" && Date.parse(t.at) > since)),
  );
  return last?.kind === "observe" && last.silent;
}

/**
 * The plan, one card per step, the same list in a draft and once started. Each step has its
 * own play control and runs only itself; each result sits under the step that made it.
 *
 * The page draws the heading, Start and "Run to the next stop" above this, and the study's
 * conversation beside it: `onTalk` hands it the step a message should be about.
 */
export function StudySteps({
  studyId,
  onTalk,
}: {
  studyId: string;
  onTalk: (stepId: string) => void;
}) {
  const { data: study } = useStudy(studyId);
  const { data: thread } = useThread(studyId);
  const run = useRunStep(studyId);
  const swap = useSwapTool(studyId);
  const approve = useApprove(studyId);

  if (!study) return null;
  if (study.steps.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No steps yet. Daikon Studio writes the plan once it has asked what it needs to know.
      </p>
    );
  }

  const started = study.status !== "draft";
  const busy = study.running || run.isPending || swap.isPending;
  const turns = thread?.turns ?? [];
  const choosing = study.running
    ? RUN_UNDER_WAY
    : approve.isPending
      ? "Recording your choice."
      : null;

  return (
    <ol className="space-y-2">
      {study.steps.map((step) => (
        <StepCard
          key={step.id}
          studyId={study.id}
          step={step}
          started={started}
          disabled={busy}
          looked={quiet(step, turns)}
          onRun={() => run.mutate(step.id)}
          onSwap={(toolId) => swap.mutate({ stepId: step.id, toolId })}
          onApprove={
            started && step.judgment && step.status === "done"
              ? (candidate) => approve.mutate({ stepId: step.id, candidate })
              : undefined
          }
          approveBlocked={choosing}
          onTalk={() => onTalk(step.id)}
        />
      ))}
    </ol>
  );
}
