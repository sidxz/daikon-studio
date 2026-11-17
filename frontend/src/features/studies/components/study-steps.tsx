"use client";

import { useApprove, useRunStep, useStudy, useSwapTool, useThread } from "../hooks/use-studies";
import { RUN_UNDER_WAY, type StudyStep, type Turn } from "../types";
import { StepCard } from "./step-card";

/**
 * The agent's newest look at a step's current result, or undefined if it has not looked yet.
 * Only a turn made after the step last started is about that result; an earlier one belongs
 * to the result this run replaced.
 */
function observation(step: StudyStep, turns: Turn[]): Turn | undefined {
  if (!step.started_at) return undefined;
  const since = Date.parse(step.started_at);
  return turns.findLast(
    (t) =>
      t.about === step.id &&
      (t.kind === "observe" || t.kind === "observe_failed") &&
      Date.parse(t.at) > since,
  );
}

/**
 * Whether the agent's newest look at a step's current result found nothing to add. A failure
 * to observe means nobody looked, so it is not quiet.
 */
function quiet(step: StudyStep, turns: Turn[]): boolean {
  if (step.status !== "done" && step.status !== "failed") return false;
  const last = observation(step, turns);
  return last?.kind === "observe" && last.silent;
}

/**
 * Whether the step's tool has finished but the agent has not read the result yet.
 *
 * The run holds the study for this whole window -- the tool and then the look at what it made
 * -- so nothing else can run during it. Without this the card drops "Running" the moment the
 * tool returns, and every play button on the page is disabled with nothing saying why.
 */
function reading(step: StudyStep, turns: Turn[], running: boolean): boolean {
  if (!running || (step.status !== "done" && step.status !== "failed")) return false;
  return observation(step, turns) === undefined;
}

/**
 * The plan, one card per step, the same list in a draft and once started. Each step has its
 * own play control and runs only itself; each result sits under the step that made it.
 *
 * The page draws the heading, Start and "Continue execution" above this, and the study's
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
        The analysis plan will be prepared after the initial clarification stage.
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
          reading={reading(step, turns, study.running)}
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
