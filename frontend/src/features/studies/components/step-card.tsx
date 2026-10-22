import { StatusDot } from "@/shared/components/status-dot";
import { Button } from "@/shared/components/ui/button";
import { Loader2, MessageSquare, PauseCircle, Play } from "lucide-react";
import { STEP_STATUS_COPY, STEP_TONE, type StudyStep } from "../types";
import { ValueBlock } from "./value-block";

function param(value: unknown): string {
  return typeof value === "object" && value !== null ? JSON.stringify(value) : String(value);
}

/** One step, with the one control that runs it: this step, never the ones after. */
export function StepCard({
  studyId,
  step,
  disabled,
  onRun,
  onTalk,
  talking,
}: {
  studyId: string;
  step: StudyStep;
  /** True while any step of the study is running: one run at a time. */
  disabled: boolean;
  onRun: () => void;
  /** Focus this step's conversation in the panel beside the results. */
  onTalk: () => void;
  talking: boolean;
}) {
  const params = Object.entries(step.params);
  const label = step.status === "done" ? "Run this step again" : "Run this step";
  return (
    <li className="min-w-0 space-y-3 rounded-sm border p-3">
      <div className="flex items-start gap-2">
        <span className="mt-2">
          <StatusDot tone={STEP_TONE[step.status] ?? "muted"} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium">{step.name}</p>
          <p className="text-xs text-muted-foreground">
            {STEP_STATUS_COPY[step.status] ?? step.status}
          </p>
        </div>
        {step.gated && (
          <span className="mt-2 flex items-center gap-1 whitespace-nowrap text-xs text-muted-foreground">
            <PauseCircle className="size-3.5" aria-hidden="true" />
            stops for you
          </span>
        )}
        {step.status === "running" ? (
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
        )}
      </div>

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

      {step.error && (
        <div
          className="rounded-sm border border-destructive/40 bg-destructive/5 p-3 text-sm"
          role="alert"
        >
          {step.error}
        </div>
      )}

      {step.outputs.map((value) => (
        <ValueBlock key={value.id} studyId={studyId} value={value} />
      ))}

      {step.discussable && (
        <Button
          variant="ghost"
          size="sm"
          className="px-0 text-muted-foreground hover:bg-transparent hover:text-foreground"
          aria-pressed={talking}
          onClick={onTalk}
        >
          <MessageSquare className="size-4" aria-hidden="true" />
          {/* One label, always. A participle reads as a status, and a button
              that describes a state looks broken when pressing it changes
              nothing visible. Pressed-ness is for aria and the styling. */}
          Talk about this result
        </Button>
      )}
    </li>
  );
}
