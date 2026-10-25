"use client";

import { cn } from "@/shared/lib/utils";
import { useStudy, useThread } from "../hooks/use-studies";
import type { Turn } from "../types";

/**
 * Evaluation criteria, and when the plan would stop instead, as the study holds them,
 * each with where it stands now.
 *
 * The standing lives here and not in the thread. The agent reports a criterion's state as it
 * goes, and with nowhere to accumulate it, every turn restated the ones that had not moved,
 * dragging each criterion's full wording along with it. Here the latest word on a criterion
 * sits beside the criterion, so the thread can say only what changed.
 *
 * There is no "3 of 5" and there never should be: the note above says the answer cannot be
 * decided after seeing the data, and a score to push over the line is exactly the pressure
 * writing the criteria in advance exists to remove. A state can also go backwards, because a
 * later step can unmake what an earlier one showed.
 */
export function PlanTerms({
  studyId,
  checks,
  stopConditions,
  amendedAfterSteps = null,
}: {
  studyId: string;
  checks: string[];
  stopConditions: string[];
  /** How many steps had run when the checks or stop conditions last changed; null if never. */
  amendedAfterSteps?: number | null;
}) {
  const thread = useThread(studyId);
  const { data: study } = useStudy(studyId);
  const turns = thread.data?.turns ?? [];
  const steps = study?.steps ?? [];

  /** The latest word on one criterion, from the newest turn that pointed at it. */
  const standing = (evidence: "check" | "stop_condition", text: string) => {
    for (let i = turns.length - 1; i >= 0; i--) {
      const point = turns[i].points.find(
        (p) =>
          p.evidence === evidence &&
          p.cites.some((c) => c.trim().toLowerCase() === text.trim().toLowerCase()),
      );
      if (point) {
        const step = steps.find((s) => s.id === turns[i].about);
        return {
          label: point.label,
          state: point.state ?? "",
          at: step ? `step ${step.position + 1}` : null,
        };
      }
    }
    return null;
  };

  return (
    <>
      {checks.length > 0 && (
        <section className="space-y-2">
          <h2 className="text-sm font-medium">Evaluation criteria</h2>
          {amendedAfterSteps === null && (
            <p className="text-xs text-muted-foreground">
              Criteria are specified before execution to reduce the risk of interpreting results
              against retrospectively selected criteria.
            </p>
          )}
          <ul className="space-y-2 text-sm">
            {checks.map((check) => (
              <Term key={check} text={check} standing={standing("check", check)} />
            ))}
          </ul>
        </section>
      )}

      {stopConditions.length > 0 && (
        <section className="space-y-2">
          <h2 className="text-sm font-medium">Stopping criteria</h2>
          <ul className="space-y-2 text-sm">
            {stopConditions.map((condition) => (
              <Term
                key={condition}
                text={condition}
                standing={standing("stop_condition", condition)}
              />
            ))}
          </ul>
        </section>
      )}

      {amendedAfterSteps !== null && (
        <p className="text-xs text-muted-foreground">
          {amendedAfterSteps === 0
            ? "Amended before any step ran."
            : `Amended after ${amendedAfterSteps} ${amendedAfterSteps === 1 ? "step" : "steps"} had run.`}
        </p>
      )}
    </>
  );
}

/**
 * The mark beside a criterion, from the standing the agent declared rather than read out of
 * its sentence. It used to be guessed from the wording, which could not tell "met" from "met
 * for one detector, not the other", and a wrong tick is the one mistake here a reader acts on
 * without reading further. A turn written before the state existed has none, and shows the
 * neutral mark with its reason, which is what it always was.
 */
const MARK: Record<string, { glyph: string; className: string }> = {
  met: { glyph: "✓", className: "text-emerald-500" },
  not_met: { glyph: "✕", className: "text-destructive" },
  not_testable: { glyph: "!", className: "text-amber-500" },
  "": { glyph: "·", className: "text-muted-foreground" },
};

function Term({
  text,
  standing,
}: {
  text: string;
  standing: { label: string; state: string; at: string | null } | null;
}) {
  const mark = MARK[standing?.state ?? ""] ?? MARK[""];
  return (
    <li className="grid grid-cols-[1rem_1fr] gap-x-2">
      <span className={cn("select-none", mark.className)} aria-hidden>
        {mark.glyph}
      </span>
      <div className="min-w-0 space-y-0.5">
        <p className="text-muted-foreground">{text}</p>
        {standing && (
          <p className={cn("text-xs", mark.className)}>
            {standing.label}
            {standing.at && <span className="text-muted-foreground"> · {standing.at}</span>}
          </p>
        )}
      </div>
    </li>
  );
}
