"use client";

import { Checkbox } from "@/shared/components/ui/checkbox";
import { Input } from "@/shared/components/ui/input";
import { Label } from "@/shared/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/components/ui/select";
import { hasRandomComparison, isReplicableSplit } from "@/shared/lib/split";

export interface EvaluationOptions {
  /** A dataset column whose true rows are scored again on their own; "" for none. */
  subsetColumn: string;
  runBaseline: boolean;
  optimismGap: boolean;
  splitReplicates: number;
}

export const DEFAULT_EVALUATION_OPTIONS: EvaluationOptions = {
  subsetColumn: "",
  runBaseline: true,
  optimismGap: true,
  splitReplicates: 0,
};

/** Whether anything here differs from what a protocol does when left alone. */
export function evaluationOptionsChanged(value: EvaluationOptions): boolean {
  return (
    !value.runBaseline ||
    !value.optimismGap ||
    value.subsetColumn !== "" ||
    value.splitReplicates > 0
  );
}

/**
 * How a run is evaluated, as opposed to how its model is trained.
 *
 * Shared by the protocol form and the sweep form rather than written twice: a sweep is
 * several protocols measured against one another, so an option one form offers and the
 * other does not is a comparison a scientist can only build halfway. It is also the
 * difference between a sweep costing one fit per configuration and three, which is what
 * makes sweeping settings usable as a quick read rather than an overnight job.
 *
 * Each control hides itself where it would mean nothing: a split that has no random
 * comparison cannot report an optimism gap, one that cannot be redrawn cannot take
 * replicates, and a dataset with no spare columns has nothing to flag a subset with.
 */
export function EvaluationOptionsFields({
  value,
  onChange,
  columns,
  splitStrategy,
  idPrefix = "evaluation",
}: {
  value: EvaluationOptions;
  onChange: (patch: Partial<EvaluationOptions>) => void;
  /** Dataset columns eligible to flag a subset; undefined until a dataset is chosen. */
  columns?: string[];
  /** The dataset's split strategy; undefined until a dataset is chosen. */
  splitStrategy?: string;
  idPrefix?: string;
}) {
  return (
    <div className="space-y-3">
      {columns != null && columns.length > 0 && (
        <div className="space-y-1.5 rounded-lg border p-4">
          <Label htmlFor={`${idPrefix}-subset-column`} className="font-normal">
            Report a separate metric for one group of compounds
          </Label>
          <Select
            value={value.subsetColumn}
            onValueChange={(next) => onChange({ subsetColumn: next })}
          >
            <SelectTrigger id={`${idPrefix}-subset-column`}>
              <SelectValue placeholder="None" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="">None</SelectItem>
              {columns.map((column) => (
                <SelectItem key={column} value={column}>
                  {column}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Rows where this column is true are measured again on their own, against the baseline
            over the same rows. Use it when the number you are comparing against was reported over
            part of a test set rather than all of it.
          </p>
        </div>
      )}
      <div className="space-y-3 rounded-lg border p-4">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <Checkbox
              id={`${idPrefix}-run-baseline`}
              checked={value.runBaseline}
              onCheckedChange={(next) => onChange({ runBaseline: next !== false })}
            />
            <Label htmlFor={`${idPrefix}-run-baseline`} className="font-normal">
              Compare against a baseline model
            </Label>
          </div>
          <p className="text-xs text-muted-foreground">
            On by default. Fits a reference model for the scorecard comparison. Turning it off
            removes the baseline comparison and verdict, and is worth doing when you already know
            what the reference scores.
          </p>
        </div>
        {splitStrategy != null && hasRandomComparison(splitStrategy) && (
          <div className="space-y-1.5 border-t pt-3">
            <div className="flex items-center gap-2">
              <Checkbox
                id={`${idPrefix}-optimism-gap`}
                checked={value.optimismGap}
                onCheckedChange={(next) => onChange({ optimismGap: next !== false })}
              />
              <Label htmlFor={`${idPrefix}-optimism-gap`} className="font-normal">
                Also train on a random split
              </Label>
            </div>
            <p className="text-xs text-muted-foreground">
              On by default. Adds training on a random split so the scorecard can report the
              optimism gap. Turning it off reduces training time and removes that comparison.
            </p>
          </div>
        )}
        {splitStrategy != null && isReplicableSplit(splitStrategy) && (
          <div className="space-y-1.5 border-t pt-3">
            <Label htmlFor={`${idPrefix}-split-replicates`} className="font-normal">
              Extra draws of the split
            </Label>
            <Input
              id={`${idPrefix}-split-replicates`}
              type="number"
              min={0}
              max={10}
              value={value.splitReplicates}
              onChange={(event) =>
                // Clamped here as well as on the server, so a plan summary can never
                // promise work the request would be refused for.
                onChange({
                  splitReplicates: Math.max(0, Math.min(10, Number(event.target.value) || 0)),
                })
              }
              className="w-24"
            />
            <p className="text-xs text-muted-foreground">
              None by default. Each draw creates another training and test split of the same
              compounds and adds a full training run. The scorecard reports score variation across
              splits to help assess whether a lead over the baseline is consistent.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
