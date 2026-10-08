/**
 * Pasted-table parsing for the `chart` embed. Charts store the author's pasted
 * text verbatim (see nodes/chart.ts) and parse it at render, so this is the one
 * place that decides what a column is and which columns get plotted.
 *
 * ponytail: no quoted-field handling. Instrument exports and pandas `to_csv`
 * output for this domain don't put the delimiter inside a cell, and the tab
 * sniff below already covers the common "label contains a comma" case. Add a
 * real CSV reader (papaparse) the first time someone hits a quoted field.
 */

export type TableRow = Record<string, string | number>;
export type ParsedTable = { columns: string[]; rows: TableRow[] };

/** Fixed hue slots — a 6th series would have to reuse a hue, so callers cap here. */
export const MAX_SERIES = 5;

/** A cell is a number only if the whole trimmed cell parses; `Number("")` is 0,
 *  so the empty check has to come first or every blank becomes a zero. */
function coerce(cell: string): string | number {
  const t = cell.trim();
  if (!t) return "";
  const n = Number(t);
  return Number.isFinite(n) ? n : t;
}

export function parseTable(raw: string): ParsedTable {
  const lines = raw.split(/\r?\n/).filter((l) => l.trim() !== "");
  if (lines.length === 0) return { columns: [], rows: [] };

  // Tab wins whenever one appears anywhere: a tab-delimited paste (Excel,
  // pandas `to_csv(sep="\t")`) may legitimately contain commas inside a label,
  // but a comma-delimited paste never contains a tab.
  const delimiter = lines.some((l) => l.includes("\t")) ? "\t" : ",";
  const split = (line: string) => line.split(delimiter).map((c) => c.trim());

  const columns = split(lines[0]);
  const rows = lines.slice(1).map((line) => {
    const cells = split(line);
    // Ragged rows pad rather than drop: a truncated trailing column is far more
    // likely to be a missing measurement than a reason to discard the row.
    return Object.fromEntries(columns.map((c, i) => [c, coerce(cells[i] ?? "")])) as TableRow;
  });

  return { columns, rows };
}

// Strict ISO only. `01/02/2026` is deliberately NOT accepted: day-first and
// month-first are both in common use and guessing wrong silently moves a point
// by up to eleven months. Anything unrecognised stays a plain category, which
// is the behaviour these columns already had.
const ISO_DATE = /^(\d{4})-(\d{2})-(\d{2})$/;
const ISO_DATETIME = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$/;

/**
 * Epoch milliseconds for an ISO date cell, or null if it isn't one.
 *
 * A date-only cell is read as LOCAL midnight, not UTC. `new Date("2026-01-15")`
 * is UTC midnight, which formats as the 14th anywhere west of Greenwich — the
 * classic off-by-one-day that makes a time axis quietly disagree with the table
 * underneath it.
 */
export function parseDateCell(value: string | number): number | null {
  if (typeof value !== "string") return null;
  const text = value.trim();

  const parts = ISO_DATE.exec(text);
  if (parts) {
    const [y, m, d] = [Number(parts[1]), Number(parts[2]), Number(parts[3])];
    const date = new Date(y, m - 1, d);
    // Date rolls over silently: new Date(2026, 12, 45) is a real date in 2027.
    // Round-trip the parts so "2026-13-45" is rejected rather than misplaced.
    const valid = date.getFullYear() === y && date.getMonth() === m - 1 && date.getDate() === d;
    return valid ? date.getTime() : null;
  }

  if (ISO_DATETIME.test(text)) {
    const ms = new Date(text.replace(" ", "T")).getTime();
    return Number.isFinite(ms) ? ms : null;
  }
  return null;
}

/** True when every non-empty cell is an ISO date — the signal to give the axis
 *  a real time scale instead of spacing the rows evenly as categories. */
export function isDateColumn(table: ParsedTable, column: string): boolean {
  const present = table.rows.map((r) => r[column]).filter((v) => v !== "");
  return present.length > 0 && present.every((v) => parseDateCell(v) !== null);
}

export function isNumericColumn(table: ParsedTable, column: string): boolean {
  const present = table.rows.map((r) => r[column]).filter((v) => v !== "");
  return present.length > 0 && present.every((v) => typeof v === "number");
}

/**
 * Turn the node's stored (possibly stale or absent) column choices into ones
 * that exist in the current data. Stored names are dropped rather than trusted
 * because the author can edit the pasted table without revisiting the column
 * pickers, and a chart pointed at a deleted column should fall back, not break.
 */
export function resolveColumns(
  table: ParsedTable,
  x: string | null,
  series: string[] | null,
): { x: string; series: string[] } {
  if (table.columns.length === 0) return { x: "", series: [] };

  const resolvedX = x && table.columns.includes(x) ? x : table.columns[0];

  const chosen = series?.filter((c) => table.columns.includes(c) && c !== resolvedX);
  const resolvedSeries =
    chosen && chosen.length > 0
      ? chosen
      : table.columns.filter((c) => c !== resolvedX && isNumericColumn(table, c));

  return { x: resolvedX, series: resolvedSeries.slice(0, MAX_SERIES) };
}
