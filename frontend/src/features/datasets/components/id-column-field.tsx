"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/components/ui/select";
import type { DatasetResponse } from "@/shared/lib/api/model";
import { useState } from "react";
import { useDatasetColumns, useSetDatasetIdColumn } from "../hooks/use-datasets";

const NONE = "__none__";

/** Which column holds the compounds' own IDs; any editor may change it. */
export function IdColumnField({ dataset }: { dataset: DatasetResponse }) {
  const [open, setOpen] = useState(false);
  const columns = useDatasetColumns(dataset.id, dataset.can_edit && open);
  const save = useSetDatasetIdColumn();

  if (!dataset.can_edit) {
    return <span className="font-mono">{dataset.id_column ?? "None"}</span>;
  }
  const options = columns.data?.columns ?? (dataset.id_column ? [dataset.id_column] : []);
  return (
    <Select
      value={dataset.id_column ?? NONE}
      onOpenChange={setOpen}
      disabled={save.isPending}
      onValueChange={(value) =>
        save.mutate({ id: dataset.id, idColumn: value === NONE ? null : value })
      }
    >
      <SelectTrigger aria-label="Identifier column" className="h-8 w-48 font-mono text-xs">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={NONE}>None</SelectItem>
        {options.map((column) => (
          <SelectItem key={column} value={column} className="font-mono">
            {column}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
