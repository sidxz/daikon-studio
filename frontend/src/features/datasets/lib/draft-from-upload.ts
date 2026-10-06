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

export type ColumnRole = "structure" | "identifier" | "target" | "unused";

export function columnRole(draft: DatasetDraft, column: string): ColumnRole {
  if (draft.structureColumn === column) return "structure";
  if (draft.idColumn === column) return "identifier";
  return draft.targets.some((target) => target.column === column) ? "target" : "unused";
}

/** A role change is exclusive; it never discards the settings of another target. */
export function setColumnRole(
  draft: DatasetDraft,
  column: string,
  role: ColumnRole,
  rows: Record<string, string>[],
): DatasetDraft {
  const target = draft.targets.find((candidate) => candidate.column === column);
  const next = {
    ...draft,
    structureColumn: draft.structureColumn === column ? "" : draft.structureColumn,
    idColumn: draft.idColumn === column ? null : draft.idColumn,
    targets: draft.targets.filter((candidate) => candidate.column !== column),
  };
  if (role === "structure") next.structureColumn = column;
  if (role === "identifier") next.idColumn = column;
  if (role === "target") next.targets = [...next.targets, target ?? draftTarget(column, rows)];
  return next;
}

/** Replacing an upload keeps compatible mappings and scientist-entered metadata. */
export function replaceUpload(
  draft: DatasetDraft,
  columns: string[],
  rows: Record<string, string>[],
  file: File,
): DatasetDraft {
  const guessed = draftFromUpload(columns, rows, file.name);
  const structureColumn = columns.includes(draft.structureColumn)
    ? draft.structureColumn
    : guessed.structureColumn;
  const targets = draft.targets.filter(
    (target) => columns.includes(target.column) && target.column !== structureColumn,
  );
  const idColumn =
    draft.idColumn && columns.includes(draft.idColumn) ? draft.idColumn : guessed.idColumn;
  return withColumns(
    { ...draft, file, name: draft.name || guessed.name, structureColumn, targets, idColumn },
    {},
  );
}
