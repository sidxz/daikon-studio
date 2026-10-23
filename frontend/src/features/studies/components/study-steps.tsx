"use client";

import { useApprove, useRunStep, useStudy, useSwapTool, useThread } from "../hooks/use-studies";
import { StepCard } from "./step-card";

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
  // The newest observation of each step; one with nothing to say is a tick on its card.
  const observed = new Map(
    (thread?.turns ?? []).filter((t) => t.kind === "observe").map((t) => [t.about, t.silent]),
  );

  return (
    <ol className="space-y-2">
      {study.steps.map((step) => (
        <StepCard
          key={step.id}
          studyId={study.id}
          step={step}
          started={started}
          disabled={busy}
          looked={
            (step.status === "done" || step.status === "failed") && observed.get(step.id) === true
          }
          onRun={() => run.mutate(step.id)}
          onSwap={(toolId) => swap.mutate({ stepId: step.id, toolId })}
          onApprove={
            started && step.judgment && step.status === "done"
              ? (candidate) => approve.mutate({ stepId: step.id, candidate })
              : undefined
          }
          onTalk={() => onTalk(step.id)}
        />
      ))}
    </ol>
  );
}
