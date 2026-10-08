import { describe, expect, it } from "vitest";

import {
  isDateColumn,
  isNumericColumn,
  parseDateCell,
  parseTable,
  resolveColumns,
} from "./parse-table";

describe("parseTable", () => {
  it("parses comma-delimited text with a header row", () => {
    const t = parseTable("compound,ic50\nA,12.5\nB,3");
    expect(t.columns).toEqual(["compound", "ic50"]);
    expect(t.rows).toEqual([
      { compound: "A", ic50: 12.5 },
      { compound: "B", ic50: 3 },
    ]);
  });

  it("prefers tabs when any line contains one (Excel paste)", () => {
    // The label legitimately contains a comma, so comma-splitting would be wrong.
    const t = parseTable("gene\tlog2fc\nBRCA1, isoform 2\t-1.4");
    expect(t.columns).toEqual(["gene", "log2fc"]);
    expect(t.rows).toEqual([{ gene: "BRCA1, isoform 2", log2fc: -1.4 }]);
  });

  it("keeps non-numeric cells as strings and coerces numerics, including exponents", () => {
    const t = parseTable("id,p\nx,1e-8\ny,n/a");
    expect(t.rows).toEqual([
      { id: "x", p: 1e-8 },
      { id: "y", p: "n/a" },
    ]);
  });

  it("skips blank lines and pads ragged rows with empty strings", () => {
    const t = parseTable("a,b\n\n1\n2,3\n");
    expect(t.rows).toEqual([
      { a: 1, b: "" },
      { a: 2, b: 3 },
    ]);
  });

  it("returns an empty table for blank input rather than throwing", () => {
    expect(parseTable("   ")).toEqual({ columns: [], rows: [] });
  });
});

describe("isNumericColumn", () => {
  it("is true only when every non-empty cell is a number", () => {
    const t = parseTable("label,good,bad\na,1,1\nb,2,x");
    expect(isNumericColumn(t, "good")).toBe(true);
    expect(isNumericColumn(t, "bad")).toBe(false);
    expect(isNumericColumn(t, "label")).toBe(false);
  });

  it("is false for a column with no numeric cells at all", () => {
    expect(isNumericColumn(parseTable("a\n\n"), "a")).toBe(false);
  });
});

describe("resolveColumns", () => {
  it("defaults x to the first column and series to the remaining numeric ones", () => {
    const t = parseTable("compound,ic50,note\nA,1,ok\nB,2,ok");
    expect(resolveColumns(t, null, null)).toEqual({ x: "compound", series: ["ic50"] });
  });

  it("honours explicit choices", () => {
    const t = parseTable("a,b,c\n1,2,3");
    expect(resolveColumns(t, "b", ["c"])).toEqual({ x: "b", series: ["c"] });
  });

  it("drops stored columns that no longer exist after the data was edited", () => {
    const t = parseTable("a,b\n1,2");
    expect(resolveColumns(t, "gone", ["b", "alsogone"])).toEqual({ x: "a", series: ["b"] });
  });

  it("caps the defaulted series at five so hues are never cycled", () => {
    const t = parseTable("x,a,b,c,d,e,f\n0,1,2,3,4,5,6");
    expect(resolveColumns(t, null, null).series).toEqual(["a", "b", "c", "d", "e"]);
  });

  it("returns empty roles for an empty table", () => {
    expect(resolveColumns({ columns: [], rows: [] }, null, null)).toEqual({ x: "", series: [] });
  });
});

describe("parseDateCell", () => {
  it("reads an ISO date as local midnight, not UTC", () => {
    const ms = parseDateCell("2026-01-15");
    expect(ms).not.toBeNull();
    const d = new Date(ms!);
    // The whole point: west of Greenwich, UTC midnight would render as the 14th.
    expect([d.getFullYear(), d.getMonth(), d.getDate()]).toEqual([2026, 0, 15]);
  });

  it("accepts ISO date-times, with or without a zone", () => {
    expect(parseDateCell("2026-01-15T09:30")).not.toBeNull();
    expect(parseDateCell("2026-01-15 09:30:00")).not.toBeNull();
    expect(parseDateCell("2026-01-15T09:30:00Z")).not.toBeNull();
  });

  it("rejects day/month-ambiguous formats rather than guessing", () => {
    expect(parseDateCell("01/02/2026")).toBeNull();
    expect(parseDateCell("15 Jan 2026")).toBeNull();
    expect(parseDateCell("Jan 2026")).toBeNull();
  });

  it("rejects impossible dates instead of letting them roll over", () => {
    // new Date(2026, 12, 45) is a real date in 2027 — silently the wrong point.
    expect(parseDateCell("2026-13-45")).toBeNull();
    expect(parseDateCell("2026-02-30")).toBeNull();
  });

  it("is null for numbers and plain text", () => {
    expect(parseDateCell(2026)).toBeNull();
    expect(parseDateCell("Catalyst_A")).toBeNull();
  });
});

describe("isDateColumn", () => {
  it("is true only when every non-empty cell parses", () => {
    const good = parseTable("day,titre\n2026-01-15,4.2\n2026-02-01,5.1");
    expect(isDateColumn(good, "day")).toBe(true);
    expect(isDateColumn(good, "titre")).toBe(false);

    const mixed = parseTable("day,titre\n2026-01-15,4.2\nweek 2,5.1");
    expect(isDateColumn(mixed, "day")).toBe(false);
  });

  it("tolerates blanks but is false for an all-blank column", () => {
    expect(isDateColumn(parseTable("day,v\n2026-01-15,1\n,2"), "day")).toBe(true);
    expect(isDateColumn(parseTable("day,v\n,1\n,2"), "day")).toBe(false);
  });
});
