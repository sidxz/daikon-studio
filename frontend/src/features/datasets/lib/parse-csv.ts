import Papa from "papaparse";

export interface CsvPreview {
  columns: string[];
  rows: Record<string, string>[];
}

/**
 * How many rows the column guesses get to see. The preview *table* shows four
 * of them; the rest are evidence.
 *
 * It is this large because of sparse multi-task files, where a column is blank
 * for most rows and the measured ones are not evenly spread: a 20-row sample
 * of the Tox21 challenge set is entirely blank for several endpoints, so the
 * kind was guessed from no values at all and a 0/1 endpoint opened as
 * "Continuous values". A column still blank across this many rows is one the
 * backend refuses anyway, as a target nothing was measured for.
 */
const SAMPLE_ROWS = 1000;

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
      preview: SAMPLE_ROWS,
      complete: (result) => {
        if (result.errors.length > 0) {
          reject(
            new Error(
              `Could not read this CSV: ${result.errors[0].message}. Check the delimiters and quoted values.`,
            ),
          );
          return;
        }
        const columns = (result.meta.fields ?? []).filter((field) => field.trim() !== "");
        if (columns.length === 0) {
          reject(new Error("No columns found. The file must be a CSV with a header row."));
          return;
        }
        const rows = result.data;
        if (rows.length === 0) {
          reject(new Error("The file has column headers but no data rows."));
          return;
        }
        resolve({ columns, rows });
      },
      error: (error) => reject(error),
    });
  });
}

/**
 * A column looks like a binary label when it has at least one value and every
 * value is a 0 or a 1 (so "0", "1", "0.0" and "1.0" all count). A rare label
 * can show a single value in the preview rows, so two distinct values is not
 * required. Used only to preselect the target kind -- the backend is what
 * actually decides, and it rejects the file if this guess was wrong.
 */
export function looksBinary(rows: Record<string, string>[], column: string): boolean {
  let any = false;
  for (const row of rows) {
    const value = row[column]?.trim();
    if (value === undefined || value === "") continue;
    const number = Number(value);
    if (number !== 0 && number !== 1) return false;
    any = true;
  }
  return any;
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
