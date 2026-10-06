"use client";

import type { Engine } from "@/features/engines";
import { TASK_LABELS, trainingKind } from "@/features/engines/types";
import { Badge } from "@/shared/components/ui/badge";
import { RadioGroup, RadioGroupItem } from "@/shared/components/ui/radio-group";
import { Cpu, Sparkles } from "lucide-react";

export function computeLabel(engine: Engine): string {
  return engine.lane === "gpu"
    ? "GPU lane"
    : engine.lane === "default"
      ? "CPU lane"
      : engine.lane
        ? `${engine.lane} lane`
        : "Compute lane unspecified";
}

export function EnginePicker({
  engines,
  eligible,
  value,
  onChange,
  targetCount,
}: {
  engines: Engine[];
  eligible: Engine[];
  value: string;
  onChange: (id: string) => void;
  targetCount: number;
}) {
  return (
    <RadioGroup
      aria-label="Choose an engine"
      value={value}
      onValueChange={onChange}
      className="grid gap-3 sm:grid-cols-2"
    >
      {engines.map((engine) => {
        const compatible = eligible.some((candidate) => candidate.id === engine.id);
        return (
          <label
            key={engine.id}
            htmlFor={`engine-${engine.id}`}
            className={`flex h-full flex-col gap-3 rounded-xl border p-4 transition-colors ${!compatible ? "cursor-not-allowed bg-muted/20 opacity-60" : value === engine.id ? "cursor-pointer border-primary bg-primary/5" : "cursor-pointer border-border hover:bg-muted/30"}`}
          >
            <div className="flex items-start gap-2.5">
              <RadioGroupItem
                id={`engine-${engine.id}`}
                aria-label={engine.name}
                aria-describedby={`engine-description-${engine.id}`}
                value={engine.id}
                disabled={!compatible}
                className="mt-0.5"
              />
              <span className="text-sm font-medium">{engine.name}</span>
            </div>
            <p
              id={`engine-description-${engine.id}`}
              className="text-xs leading-relaxed text-muted-foreground"
            >
              {engine.description}
            </p>
            <div className="flex flex-wrap gap-1.5">
              {engine.tasks.map((task) => (
                <Badge key={task} variant="outline" className="text-[10px] font-normal">
                  {TASK_LABELS[task] ?? task}
                </Badge>
              ))}
            </div>
            <div className="mt-auto space-y-2">
              <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Cpu className="size-3" />
                {computeLabel(engine)}
              </p>
              {targetCount > 1 && (
                <p className="text-xs text-muted-foreground">{trainingKind(engine, targetCount)}</p>
              )}
              {engine.is_baseline && (
                <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Sparkles className="size-3" />
                  Default comparison baseline
                </p>
              )}
              {!compatible && (
                <p className="text-xs">
                  Unavailable: this engine cannot train all selected target types together.
                </p>
              )}
            </div>
          </label>
        );
      })}
    </RadioGroup>
  );
}
