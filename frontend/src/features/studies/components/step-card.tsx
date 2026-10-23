import { StatusDot } from "@/shared/components/status-dot";
import { Button } from "@/shared/components/ui/button";
import { Check, Loader2, MessageSquare, PauseCircle, Play } from "lucide-react";
import { STEP_STATUS_COPY, STEP_TONE, type StudyStep } from "../types";
import { ValueBlock } from "./value-block";

function param(value: unknown): string {
  return typeof value === "object" && value !== null ? JSON.stringify(value) : String(value);
}

/**
 * One step of the plan. In a draft it is a line to argue with: its tool, its settings and
 * anything that stops it. Once started it has the one control that runs it (this step, never
 * the ones after) and the results it made. Numbered from 1, as the agent names it.
 */
export function StepCard({
  studyId,
  step,
  started,
  disabled,
  looked,
  onRun,
  onSwap,
  onApprove,
  onTalk,
}: {
  studyId: string;
  step: StudyStep;
  /** False while a draft: nothing runs before Start. */
  started: boolean;
  /** True while any step of the study is running: one run at a time. */
  disabled: boolean;
  /** The agent's latest look at this result had nothing to add. */
  looked: boolean;
  onRun: () => void;
  onSwap: (toolId: string) => void;
  /** Choose one of the step's candidates; given for a finished judgment step only. */
  onApprove?: (rank: number) => void;
  /** Put this step on the composer, so the next message is about its result. */
  onTalk: () => void;
}) {
  const number = step.position + 1;
  const params = Object.entries(step.params);
  const label = step.status === "done" ? `Run step ${number} again` : `Run step ${number}`;
  const ended = step.status === "done" || step.status === "failed";
  return (
    <li className="min-w-0 space-y-3 rounded-sm border p-3">
      <div className="flex items-start gap-2">
        <span className="mt-2">
          <StatusDot tone={STEP_TONE[step.status] ?? "muted"} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm">
            <span className="text-muted-foreground">Step {number}</span>{" "}
            <span className="font-medium">{step.name}</span>
          </p>
          <p className="text-xs text-muted-foreground">
            {STEP_STATUS_COPY[step.status] ?? step.status}
            {step.expanded_from && ` · part of: ${step.expanded_from.replace(/_/g, " ")}`}
          </p>
        </div>
        {step.gated && (
          <span className="mt-2 flex items-center gap-1 whitespace-nowrap text-xs text-muted-foreground">
            <PauseCircle className="size-3.5" aria-hidden="true" />
            stops for you
          </span>
        )}
        {started &&
          (step.status === "running" ? (
            <Button variant="ghost" size="sm" disabled>
              <Loader2 className="size-4 animate-spin" aria-hidden="true" />
              Running
            </Button>
          ) : (
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={label}
              title={label}
              disabled={disabled}
              onClick={onRun}
            >
              <Play className="size-4" aria-hidden="true" />
            </Button>
          ))}
      </div>

      {step.alternatives.length > 0 && (
        <label className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          Tool
          <select
            aria-label={`Tool for step ${number}`}
            value={step.tool_id}
            disabled={disabled}
            onChange={(event) => onSwap(event.target.value)}
            className="h-8 max-w-full rounded-md border bg-background px-2 text-sm text-foreground"
          >
            <option value={step.tool_id}>{step.name}</option>
            {step.alternatives.map((other) => (
              <option key={other.tool_id} value={other.tool_id}>
                {other.name}
              </option>
            ))}
          </select>
        </label>
      )}

      {params.length > 0 && (
        <ul className="flex flex-wrap gap-1.5">
          {params.map(([name, value]) => (
            <li key={name} className="rounded-sm border px-1.5 py-0.5 font-mono text-xs">
              {name}: {param(value)}
            </li>
          ))}
        </ul>
      )}

      <p className="text-sm text-muted-foreground">{step.purpose}</p>

      {step.problem && (
        <div
          className="rounded-sm border border-destructive/40 bg-destructive/5 p-3 text-sm"
          role="alert"
        >
          {step.problem}
        </div>
      )}

      {step.error && (
        <div
          className="rounded-sm border border-destructive/40 bg-destructive/5 p-3 text-sm"
          role="alert"
        >
          {step.error}
        </div>
      )}

      {step.outputs.map((value) => (
        <ValueBlock key={value.id} studyId={studyId} value={value} onApprove={onApprove} />
      ))}

      {looked && (
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Check className="size-3.5" aria-hidden="true" />
          Looked, nothing to add
        </p>
      )}

      {started && ended && (
        <Button
          variant="ghost"
          size="sm"
          className="px-0 text-muted-foreground hover:bg-transparent hover:text-foreground"
          onClick={onTalk}
        >
          <MessageSquare className="size-4" aria-hidden="true" />
          Talk about this result
        </Button>
      )}
    </li>
  );
}
