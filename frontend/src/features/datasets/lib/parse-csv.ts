import Papa from "papaparse";

export interface CsvPreview {
  columns: string[];
  rows: Record<string, string>[];
  /** Columns whose sampled values all parse as numbers — a hint, not a rule. */
  numericColumns: Set<string>;
}

const PREVIEW_ROWS = 20;

/**
 * Read a CSV's header and a handful of rows in the browser.
 *
 * `POST /datasets/uploads` hands back only an opaque `upload_ref` -- there is no
 * column-introspection endpoint, and none is needed, because the file is
 * already sitting in the browser. Parsing here also gives the wizard a preview
 * table for free, so a scientist can see the data they are describing.
 */
export function parseCsvPreview(file: File): Promise<CsvPreview> {
  return new Promise((resolve, reject) => {
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: true,
      preview: PREVIEW_ROWS,
      complete: (result) => {
        const columns = (result.meta.fields ?? []).filter((field) => field.trim() !== "");
        if (columns.length === 0) {
          reject(new Error("No columns found. Is this a CSV with a header row?"));
          return;
        }
        const rows = result.data;
        const numericColumns = new Set(
          columns.filter((column) =>
            rows.every((row) => {
              const value = row[column]?.trim();
              return value !== undefined && value !== "" && Number.isFinite(Number(value));
            }),
          ),
        );
        resolve({ columns, rows, numericColumns });
      },
      error: (error) => reject(error),
    });
  });
}

/**
 * A column whose sampled values are exactly two distinct entries looks like a
 * binary label. Used only to preselect the target kind -- the backend is what
 * actually decides, and it rejects the file if this guess was wrong.
 */
export function looksBinary(rows: Record<string, string>[], column: string): boolean {
  const seen = new Set<string>();
  for (const row of rows) {
    const value = row[column]?.trim();
    if (value === undefined || value === "") continue;
    seen.add(value);
    if (seen.size > 2) return false;
  }
  return seen.size === 2;
}

/**
 * The structure column, guessed from its name. Every import in this suite ships
 * a template, so the common case is that this is right; when it is not, the
 * picker is one click away.
 */
const STRUCTURE_HINTS = ["smiles", "structure", "canonical_smiles", "mol", "smi"];

export function guessStructureColumn(columns: string[]): string {
  const lowered = columns.map((column) => column.toLowerCase().trim());
  for (const hint of STRUCTURE_HINTS) {
    const index = lowered.indexOf(hint);
    if (index !== -1) return columns[index];
  }
  const partial = lowered.findIndex((column) => column.includes("smiles"));
  return partial !== -1 ? columns[partial] : (columns[0] ?? "");
}

/** Header row plus two realistic examples, for the Download template button. */
export const DATASET_TEMPLATE_CSV = [
  "smiles,activity",
  "CCO,0.42",
  "c1ccc(cc1)C(=O)Nc2ccccc2,1.87",
  "",
].join("\n");

export const PREDICTION_TEMPLATE_CSV = ["smiles", "CCO", "c1ccc(cc1)C(=O)Nc2ccccc2", ""].join("\n");
