import type { IDoesFilterPassParams } from "ag-grid-community";
import { type CustomFilterProps, useGridFilter } from "ag-grid-react";
import { useCallback, useId } from "react";
import type { NumberFilter } from "../lib/result-query";
import type { TriageRow } from "../types";

export function ClassFilter({
  model,
  onModelChange,
  column,
}: CustomFilterProps<TriageRow, unknown, NumberFilter>) {
  const id = useId();
  // AG Grid treats a changed callback as a new filter. Keep it stable across
  // result loads so filtering does not repeatedly invalidate the infinite cache.
  const selectedClass = model?.filter;
  const doesFilterPass = useCallback(
    ({ node }: IDoesFilterPassParams<TriageRow>) =>
      selectedClass == null || node.data?.readouts?.[column.getColId()]?.value === selectedClass,
    [column, selectedClass],
  );
  useGridFilter({ doesFilterPass });
  return (
    <fieldset className="space-y-3 p-4 text-sm">
      <legend className="sr-only">Predicted class</legend>
      <p className="text-xs text-muted-foreground">Positive = class 1 · Negative = class 0</p>
      {[
        ["all", "All classes"],
        ["1", "Positive"],
        ["0", "Negative"],
      ].map(([value, label]) => (
        <label key={value} className="flex cursor-pointer items-center gap-2">
          <input
            type="radio"
            name={id}
            value={value}
            checked={value === (model == null ? "all" : String(model.filter))}
            onChange={() =>
              onModelChange(
                value === "all"
                  ? null
                  : { filterType: "number", type: "equals", filter: Number(value) },
              )
            }
          />
          {label}
        </label>
      ))}
    </fieldset>
  );
}
