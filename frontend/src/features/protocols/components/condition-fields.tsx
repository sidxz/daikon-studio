"use client";

import type { Condition, ConditionType } from "@/features/engines";
import { Input } from "@/shared/components/ui/input";
import { Label } from "@/shared/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/components/ui/select";
import { Switch } from "@/shared/components/ui/switch";

/**
 * The run form's inputs, rendered entirely from the engine manifest.
 *
 * `key`, `label`, `type`, `required`, `default`, `minimum`, `maximum`,
 * `options` and `help` are everything a form needs, and the labels and help
 * strings are already written as biochemist-facing copy. Nothing about any
 * engine is hardcoded, which is what makes a Protocol self-describing enough
 * to be run by someone who has never heard of the engine underneath.
 *
 * `ConditionResponse.type` is a bare string in the contract rather than a
 * closed enum, so codegen gives no union -- the cast is narrowing to the five
 * values the backend's ConditionType can emit, and the default branch renders
 * a text box rather than nothing if a sixth ever appears.
 */
export function ConditionFields({
  conditions,
  values,
  onChange,
}: {
  conditions: Condition[];
  values: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
}) {
  if (conditions.length === 0) {
    return <p className="text-sm text-muted-foreground">This engine has nothing to configure.</p>;
  }

  return (
    <div className="space-y-4">
      {conditions.map((condition) => {
        const type = condition.type as ConditionType;
        const current = values[condition.key] ?? condition.default ?? "";
        const id = `condition-${condition.key}`;

        return (
          <div key={condition.key} className="space-y-1.5">
            <Label htmlFor={id}>
              {condition.label}
              {condition.required && <span className="ml-1 text-destructive">*</span>}
            </Label>

            {type === "enum" ? (
              <Select
                value={String(current)}
                onValueChange={(value) => onChange(condition.key, value)}
              >
                <SelectTrigger id={id}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {condition.options.map((option) => (
                    <SelectItem key={option} value={option}>
                      {option}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : type === "bool" ? (
              <div className="flex h-9 items-center">
                <Switch
                  id={id}
                  checked={Boolean(current)}
                  onCheckedChange={(checked) => onChange(condition.key, checked)}
                />
              </div>
            ) : (
              <Input
                id={id}
                type={type === "integer" || type === "number" ? "number" : "text"}
                value={String(current)}
                min={condition.minimum ?? undefined}
                max={condition.maximum ?? undefined}
                step={type === "integer" ? 1 : "any"}
                onChange={(event) => {
                  const raw = event.target.value;
                  if (type === "integer" || type === "number") {
                    onChange(condition.key, raw === "" ? undefined : Number(raw));
                  } else {
                    onChange(condition.key, raw);
                  }
                }}
              />
            )}

            {condition.help && <p className="text-xs text-muted-foreground">{condition.help}</p>}
            {(condition.minimum != null || condition.maximum != null) && (
              <p className="font-mono text-xs text-muted-foreground">
                {condition.minimum ?? "−∞"} to {condition.maximum ?? "∞"}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
