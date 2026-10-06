"use client";

import type { Condition } from "@/features/engines";
import { Button } from "@/shared/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/shared/components/ui/collapsible";
import { ChevronDown, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";
import { appliesToTasks, conditionError } from "../lib/conditions";
import { ConditionFields } from "./condition-fields";

export function settingValue(spec: Condition, value: unknown): string {
  if (spec.type === "bool") return value ? "On" : "Off";
  const index = spec.options.indexOf(String(value));
  return (index >= 0 ? spec.option_labels?.[index] : undefined) ?? String(value ?? "Default");
}

export function changedSettings(
  specs: Condition[],
  values: Record<string, unknown>,
  tasks: string[] | undefined,
  pinned: Record<string, unknown>,
): Condition[] {
  return specs.filter(
    (spec) =>
      appliesToTasks(spec, tasks) &&
      !(spec.key in pinned) &&
      JSON.stringify(values[spec.key] ?? spec.default) !== JSON.stringify(spec.default),
  );
}

export function TrainingSettings({
  title,
  conditions,
  values,
  onChange,
  onReset,
  pinned,
  tasks,
  idPrefix,
  reveal = false,
}: {
  title: string;
  conditions: Condition[];
  values: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
  onReset: () => void;
  pinned: Record<string, unknown>;
  tasks: string[] | undefined;
  idPrefix: string;
  reveal?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const shown = conditions.filter((condition) => appliesToTasks(condition, tasks));
  const essentials = shown.filter(
    (condition) => condition.required || condition.type === "enum" || condition.type === "bool",
  );
  const advanced = shown.filter((condition) => !essentials.includes(condition));
  const invalid = advanced.some(
    (condition) =>
      !(condition.key in pinned) &&
      conditionError(condition, values[condition.key] ?? condition.default) !== null,
  );
  const changed = changedSettings(shown, values, tasks, pinned);
  useEffect(() => {
    if (invalid || reveal) setOpen(true);
  }, [invalid, reveal]);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-sm font-medium">{title}</p>
          <p className="text-xs text-muted-foreground">
            {changed.length
              ? `${changed.length} settings changed from defaults`
              : "Using default settings"}
            {Object.keys(pinned).length
              ? ` · ${Object.keys(pinned).length} fixed by pretrained weights`
              : ""}
          </p>
        </div>
        <Button type="button" variant="ghost" size="sm" onClick={onReset}>
          <RotateCcw className="size-3" />
          Reset settings
        </Button>
      </div>
      {essentials.length > 0 && (
        <ConditionFields
          conditions={essentials}
          values={values}
          onChange={onChange}
          pinned={pinned}
          tasks={tasks}
          idPrefix={idPrefix}
        />
      )}
      {advanced.length > 0 && (
        <Collapsible
          open={open}
          onOpenChange={setOpen}
          className={`rounded-lg border p-4 ${invalid ? "border-destructive/40" : ""}`}
        >
          <CollapsibleTrigger className="flex w-full items-center justify-between text-left text-sm font-medium">
            <span>
              Advanced settings{" "}
              <span className="font-normal text-muted-foreground">({advanced.length})</span>
              {invalid && <span className="ml-2 text-destructive">Needs attention</span>}
            </span>
            <ChevronDown className={`size-4 transition-transform ${open ? "rotate-180" : ""}`} />
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-4">
            <ConditionFields
              conditions={advanced}
              values={values}
              onChange={onChange}
              pinned={pinned}
              tasks={tasks}
              idPrefix={idPrefix}
            />
          </CollapsibleContent>
        </Collapsible>
      )}
      {shown.length === 0 && (
        <p className="text-xs text-muted-foreground">This engine has no configurable settings.</p>
      )}
    </div>
  );
}
