import { guessIdColumn } from "@/shared/lib/guess-id-column";
import { type DatasetDraft, EMPTY_DRAFT } from "../types";
import { guessStructureColumn, looksBinary } from "./parse-csv";

/** The wizard's starting draft for an upload: every column guessed from its headers. */
export function draftFromUpload(
  columns: string[],
  rows: Record<string, string>[],
  fileName: string,
): DatasetDraft {
  const structureColumn = guessStructureColumn(columns);
  const targetColumn = columns.find((column) => column !== structureColumn) ?? "";
  return {
    ...EMPTY_DRAFT,
    name: fileName.replace(/\.csv$/i, ""),
    structureColumn,
    targetColumn,
    kind: targetColumn && looksBinary(rows, targetColumn) ? "binary" : "numeric",
    idColumn: guessIdColumn(
      columns.filter((column) => column !== targetColumn),
      structureColumn,
    ),
  };
}

/** A column change that keeps the identifier distinct from the structure and target. */
export function withColumns(
  draft: DatasetDraft,
  changes: Partial<Pick<DatasetDraft, "structureColumn" | "targetColumn">>,
): DatasetDraft {
  const next = { ...draft, ...changes };
  const clash = next.idColumn === next.structureColumn || next.idColumn === next.targetColumn;
  return clash ? { ...next, idColumn: null } : next;
}
