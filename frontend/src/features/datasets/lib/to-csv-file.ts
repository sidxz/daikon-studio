import * as XLSX from "@e965/xlsx";

/** A dropped file, ready for the CSV path, plus the workbook's sheets if it had any. */
export interface Converted {
  /** What gets previewed and uploaded. Always a CSV. */
  file: File;
  /** Every sheet in the workbook, in workbook order. Empty for a CSV. */
  sheetNames: string[];
  /** The sheet this CSV came from, or null for a CSV. */
  sheet: string | null;
}

const EXCEL = /\.(xlsx|xlsm|xls)$/i;

/**
 * Whether this file goes through the workbook reader. Callers use it for the
 * size limit: a workbook is a zip, so 25 MB on disk is several hundred MB of
 * cells once the browser expands it, while a CSV costs what it weighs.
 */
export const isExcel = (name: string) => EXCEL.test(name);

/** Megabytes a dropped file may weigh, by format. */
export const sizeLimitMb = (name: string) => (isExcel(name) ? 25 : 100);

/** What the drop zone accepts, in react-dropzone's shape. */
export const ACCEPTED_UPLOADS = {
  "text/csv": [".csv"],
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
  "application/vnd.ms-excel": [".xls"],
  "application/vnd.ms-excel.sheet.macroEnabled.12": [".xlsm"],
};

/** Excel's own error cells, for the rare workbook that stores a code without its text. */
const ERRORS: Record<number, string> = {
  0: "#NULL!",
  7: "#DIV/0!",
  15: "#VALUE!",
  23: "#REF!",
  29: "#NAME?",
  36: "#NUM!",
  42: "#N/A",
  43: "#GETTING_DATA",
};

const pad = (n: number) => String(n).padStart(2, "0");

/**
 * An Excel date serial as an ISO 8601 string. The serial itself is meaningless
 * downstream, and the displayed form is whatever the author's locale wrote
 * ("10/10/26"), so neither can be handed to the backend.
 *
 * The arithmetic is SheetJS's rather than ours: it already carries Excel's 1900
 * leap-year bug. A date with no time of day stays a plain date, because that is
 * what a measurement date is.
 */
function isoDate(serial: number): string {
  const d = XLSX.SSF.parse_date_code(serial);
  if (!d) return String(serial);
  const day = `${d.y}-${pad(d.m)}-${pad(d.d)}`;
  return d.H === 0 && d.M === 0 && d.S === 0 ? day : `${day}T${pad(d.H)}:${pad(d.M)}:${pad(d.S)}`;
}

/**
 * One cell as the text the CSV will carry.
 *
 * The rule throughout is that a number keeps the precision it was *stored*
 * with, never the precision it was *displayed* with: a potency stored as
 * 7.4499998 under a two-decimal format reads "7.45" on screen, and training on
 * 7.45 would be training on the formatting. `cell.w` -- SheetJS's rendered text
 * -- is therefore used only for error cells, where the text *is* the value.
 *
 * Blank stays blank and never becomes zero: a sparse multi-task label depends
 * on the difference between "not measured" and "measured as 0".
 *
 * Errors (`#N/A`) and booleans (`TRUE`) pass through as themselves rather than
 * being coerced, because the backend's target gate already nulls a cell it
 * cannot use and reports it by row with a reason. A guess here would replace a
 * precise complaint with a silent wrong number.
 */
function cellText(cell: XLSX.CellObject | undefined): string {
  if (!cell || cell.t === "z" || cell.v === undefined || cell.v === null) return "";
  switch (cell.t) {
    case "e":
      return cell.w ?? ERRORS[Number(cell.v)] ?? "#ERROR";
    case "b":
      return cell.v ? "TRUE" : "FALSE";
    case "d":
      return cell.v instanceof Date ? cell.v.toISOString().slice(0, 10) : String(cell.v);
    case "n":
      return cell.z && XLSX.SSF.is_date(String(cell.z))
        ? isoDate(cell.v as number)
        : String(cell.v);
    default:
      return String(cell.v);
  }
}

const needsQuotes = (value: string) => /[",\n\r]/.test(value) || value.trim() !== value;

const csvCell = (value: string) =>
  needsQuotes(value) ? `"${value.replaceAll('"', '""')}"` : value;

/** The sheet as CSV text, trimmed to the rows and columns that actually hold data. */
function sheetToCsv(sheet: XLSX.WorkSheet): string {
  const ref = sheet["!ref"];
  if (!ref) return "";
  const range = XLSX.utils.decode_range(ref);
  const grid: string[][] = [];
  for (let r = range.s.r; r <= range.e.r; r++) {
    const row: string[] = [];
    for (let c = range.s.c; c <= range.e.c; c++) {
      row.push(cellText(sheet[XLSX.utils.encode_cell({ r, c })] as XLSX.CellObject | undefined));
    }
    grid.push(row);
  }
  // Excel's stored range routinely overshoots the table -- a stray format or a
  // cleared cell extends it -- so the blank edges come off here rather than
  // reaching the preview as unnamed columns and empty rows.
  while (grid.length > 0 && grid[grid.length - 1].every((value) => value === "")) grid.pop();
  const width = grid.reduce(
    (widest, row) => Math.max(widest, row.findLastIndex((value) => value !== "") + 1),
    0,
  );
  if (width === 0) return "";
  return `${grid.map((row) => row.slice(0, width).map(csvCell).join(",")).join("\n")}\n`;
}

/**
 * A dropped file as a CSV, so that Excel is handled once at the door instead of
 * everywhere after it.
 *
 * The browser previews the file and the backend ingests it. Those are already
 * two parsers of the same bytes, and teaching both of them Excel would make
 * four paths whose date and rounding behaviour would not agree -- the preview
 * would show one thing and the run would train on another. Converting here
 * means the bytes the backend reads are the bytes the preview showed.
 */
export async function toCsvFile(file: File, sheet?: string): Promise<Converted> {
  if (!EXCEL.test(file.name)) return { file, sheetNames: [], sheet: null };

  const book = XLSX.read(new Uint8Array(await file.arrayBuffer()), {
    type: "array",
    // Number formats are needed to tell a date from the number it is stored as.
    cellNF: true,
    // Cached results, not formula text: a column of `=AVERAGE(...)` is data.
    cellFormula: false,
  });

  // Converted at most once per sheet, because choosing between them means
  // measuring them.
  const csvBySheet = new Map<string, string>();
  const csvFor = (name: string) => {
    const seen = csvBySheet.get(name);
    if (seen !== undefined) return seen;
    const text = sheetToCsv(book.Sheets[name]);
    csvBySheet.set(name, text);
    return text;
  };
  const rowsIn = (name: string) => {
    const text = csvFor(name);
    return text === "" ? 0 : text.trimEnd().split("\n").length;
  };

  // The sheet with the most rows, earliest winning a tie -- not the first sheet
  // that holds anything. A supplementary workbook opens with a caption sheet,
  // and "the first sheet with data" imports that one line of prose. Row count
  // is the one signal that separates a caption from a table without guessing at
  // column counts, which would rule out the single-column sheet a prediction
  // input legitimately is. Where it picks wrong -- a long notes sheet beside a
  // short table -- the sheet picker in the wizard is the answer, which is why
  // the sheet that was used is always named back to the scientist.
  const chosen =
    sheet ??
    book.SheetNames.reduce<string | null>(
      (best, name) => (rowsIn(name) > (best === null ? 0 : rowsIn(best)) ? name : best),
      null,
    );
  if (chosen === null) {
    throw new Error(
      `${file.name} has no data in any of its sheets. Check that the measurements are in a sheet with a header row.`,
    );
  }
  const csv = csvFor(chosen);
  if (csv === "") {
    throw new Error(`The sheet "${chosen}" is empty. Choose a sheet that holds the measurements.`);
  }
  return {
    file: new File([csv], file.name.replace(EXCEL, ".csv"), { type: "text/csv" }),
    sheetNames: book.SheetNames,
    sheet: chosen,
  };
}
