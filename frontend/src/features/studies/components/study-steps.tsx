"use client";

import { QueryError } from "@/shared/components/query-error";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/components/ui/card";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { useState } from "react";
import { useRunStep, useStudy } from "../hooks/use-studies";
import { StageChatRail } from "./stage-chat";
import { StepCard } from "./step-card";
import { ValueBlock } from "./value-block";

/**
 * A plan, run one step at a time. Each step has its own play control and runs
 * only itself; each result sits under the step that made it.
 *
 * Drawn under the plan on the study's page, which already shows the goal, the
 * answers and the plan's terms. Only what the plan cannot show is here: what was
 * brought and the steps as they run.
 */
export function StudySteps({ studyId }: { studyId: string }) {
  const { data: study, isLoadingError, refetch, isRefetching } = useStudy(studyId);
  const run = useRunStep(studyId);
  const [talking, setTalking] = useState<number | null>(null);
  // Bumped on every press, not just a change of stage: pressing the stage the
  // panel already shows must still do something, or the button reads as dead.
  const [asked, setAsked] = useState(0);

  if (isLoadingError) {
    return <QueryError title="Could not load this study" retry={refetch} retrying={isRefetching} />;
  }
  if (!study) {
    return (
      <div className="w-full min-w-0 space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  const busy = study.running || run.isPending;
  // The study's own conversation leads, because where a study stands against the
  // checks it set itself is the question a reader arrives with; the stages are
  // what they drill into afterwards.
  const stages: { position: number | null; name: string }[] = [
    { position: null, name: "Whole study" },
    ...study.steps
      .filter((s) => s.discussable)
      .map((s) => ({ position: s.position, name: s.name })),
  ];
  const active = talking;

  return (
    /* No `items-start`: the panel has to be as tall as the steps beside it, or
       `sticky` has nothing to stick within and the conversation scrolls away
       from the table it is talking about, which is the one thing it must not
       do. */
    <div className="flex w-full min-w-0 flex-col gap-6 xl:flex-row">
      <div className="w-full min-w-0 space-y-6 wrap-anywhere">
        {study.brought.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>What you brought</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {study.brought.map((value) => (
                <ValueBlock key={value.id} studyId={study.id} value={value} />
              ))}
            </CardContent>
          </Card>
        )}

        <section className="space-y-2">
          <h2 className="text-sm font-medium">The steps</h2>
          <ol className="space-y-2">
            {study.steps.map((step) => (
              <StepCard
                key={step.position}
                studyId={study.id}
                step={step}
                disabled={busy}
                onRun={() => run.mutate(step.position)}
                onTalk={() => {
                  setTalking(step.position);
                  setAsked((n) => n + 1);
                }}
                talking={step.position === active}
              />
            ))}
          </ol>
          {run.error && (
            <p className="text-sm text-destructive" role="alert">
              {run.error.message}
            </p>
          )}
        </section>
      </div>

      {stages.length > 0 && (
        <aside className="w-full min-w-0 shrink-0 xl:w-[30rem]">
          {/* Pinned, so a claim can be read against the table it points at
              rather than from memory after scrolling. */}
          <div className="xl:sticky xl:top-4 xl:max-h-[calc(100vh-2rem)] xl:overflow-y-auto">
            <StageChatRail
              studyId={study.id}
              stages={stages}
              active={active}
              onSelect={setTalking}
              focusToken={asked}
            />
          </div>
        </aside>
      )}
    </div>
  );
}
