import { guessIdColumn } from "@/shared/lib/guess-id-column";
import { type DatasetDraft, type DraftTarget, EMPTY_DRAFT } from "../types";
import { guessStructureColumn, looksBinary } from "./parse-csv";

/** A newly chosen target, its kind guessed from the preview rows. */
export function draftTarget(column: string, rows: Record<string, string>[]): DraftTarget {
  return {
    column,
    kind: looksBinary(rows, column) ? "binary" : "numeric",
    unit: "",
    direction: "high",
  };
}

/** The wizard's starting draft for an upload: every column guessed from its headers. */
export function draftFromUpload(
  columns: string[],
  rows: Record<string, string>[],
  fileName: string,
): DatasetDraft {
  const structureColumn = guessStructureColumn(columns);
  // The identifier is guessed first: it is a pre-checked box, not a select the
  // real target would replace, so an `id,smiles,value` file must not open with the id checked.
  const idColumn = guessIdColumn(columns, structureColumn);
  const first = columns.find((column) => column !== structureColumn && column !== idColumn);
  return {
    ...EMPTY_DRAFT,
    name: fileName.replace(/\.csv$/i, ""),
    structureColumn,
    targets: first ? [draftTarget(first, rows)] : [],
    idColumn,
  };
}

/**
 * A column change that keeps the structure, the targets and the identifier
 * distinct: a target that becomes the structure column is dropped, and an
 * identifier that becomes either is cleared.
 */
export function withColumns(
  draft: DatasetDraft,
  changes: Partial<Pick<DatasetDraft, "structureColumn" | "targets">>,
): DatasetDraft {
  const next = { ...draft, ...changes };
  const targets = next.targets.filter((target) => target.column !== next.structureColumn);
  const clash =
    next.idColumn === next.structureColumn ||
    targets.some((target) => target.column === next.idColumn);
  return { ...next, targets, idColumn: clash ? null : next.idColumn };
}

/** Choose or unchoose one column to predict. New targets go last. */
export function toggleTarget(
  draft: DatasetDraft,
  column: string,
  chosen: boolean,
  rows: Record<string, string>[],
): DatasetDraft {
  const others = draft.targets.filter((target) => target.column !== column);
  return withColumns(draft, {
    targets: chosen ? [...others, draftTarget(column, rows)] : others,
  });
}
