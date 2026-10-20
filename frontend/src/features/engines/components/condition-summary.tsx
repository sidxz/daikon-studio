"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/shared/components/ui/table";
import { type Condition, TASK_LABELS } from "../types";

/** The label a form shows for one of a condition's option values, or the value itself. */
function optionLabel(condition: Condition, value: unknown): string {
  const index = condition.options.indexOf(String(value));
  return (index >= 0 && condition.option_labels[index]) || String(value);
}

function describeBounds(condition: Condition): string | null {
  const { minimum, maximum } = condition;
  if (minimum != null && maximum != null) return `${minimum}–${maximum}`;
  if (minimum != null) return `≥ ${minimum}`;
  if (maximum != null) return `≤ ${maximum}`;
  if (condition.options.length > 0) {
    return condition.options.map((option) => optionLabel(condition, option)).join(" · ");
  }
  return null;
}

/**
 * What an engine will let you tune, straight from its manifest.
 *
 * Every label and help string here is backend-authored, written as
 * biochemist-facing copy. Nothing about any engine is hardcoded on this side --
 * a new engine appears here the moment it registers, and the run form that
 * reuses these fields renders itself the same way.
 */
export function ConditionSummary({
  conditions,
  layout = "list",
}: { conditions: Condition[]; layout?: "list" | "table" }) {
  if (conditions.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">This engine has no configurable settings.</p>
    );
  }

  if (layout === "table") {
    return (
      <Table className="table-fixed" aria-label="Engine settings">
        <TableHeader>
          <TableRow>
            <TableHead className="w-1/2">Setting</TableHead>
            <TableHead className="w-1/5">Default</TableHead>
            <TableHead className="whitespace-normal">Allowed values</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {conditions.map((condition) => (
            <TableRow key={condition.key}>
              <TableCell className="whitespace-normal break-words py-4 align-top">
                <p className="font-medium">{condition.label}</p>
                {condition.required && (
                  <span className="text-xs text-muted-foreground">Required</span>
                )}
                {condition.help && (
                  <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                    {condition.help}
                  </p>
                )}
                {condition.tasks.length > 0 && (
                  <p className="mt-1 text-xs text-muted-foreground">
                    For {condition.tasks.map((task) => TASK_LABELS[task] ?? task).join(" · ")}
                  </p>
                )}
              </TableCell>
              <TableCell className="whitespace-normal break-words py-4 align-top text-xs">
                {condition.default == null
                  ? "Not set"
                  : condition.type === "bool"
                    ? condition.default
                      ? "Yes"
                      : "No"
                    : optionLabel(condition, condition.default)}
              </TableCell>
              <TableCell className="whitespace-normal break-words py-4 align-top text-xs text-muted-foreground">
                {describeBounds(condition) ??
                  (condition.type === "bool" ? "Yes · No" : "Not specified")}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    );
  }

  return (
    <dl className="space-y-2.5">
      {conditions.map((condition) => {
        const bounds = describeBounds(condition);
        return (
          <div key={condition.key}>
            <dt className="flex flex-wrap items-baseline gap-x-2 text-sm font-medium">
              {condition.label}
              {condition.required && (
                <span className="text-xs font-normal text-muted-foreground">required</span>
              )}
              {condition.default != null && (
                <span className="font-mono text-xs font-normal text-muted-foreground">
                  default {optionLabel(condition, condition.default)}
                </span>
              )}
              {bounds && (
                <span className="font-mono text-xs font-normal text-muted-foreground">
                  {bounds}
                </span>
              )}
            </dt>
            {condition.help && (
              <dd className="mt-0.5 text-xs text-muted-foreground">{condition.help}</dd>
            )}
          </div>
        );
      })}
    </dl>
  );
}
