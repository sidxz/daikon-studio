import * as XLSX from "@e965/xlsx";
import { describe, expect, it } from "vitest";
import { toCsvFile } from "./to-csv-file";

/** A workbook as a dropped file, built from `{ sheet: rows }`. */
function workbook(sheets: Record<string, unknown[][]>, name = "book.xlsx"): File {
  const book = XLSX.utils.book_new();
  for (const [sheet, rows] of Object.entries(sheets)) {
    XLSX.utils.book_append_sheet(book, XLSX.utils.aoa_to_sheet(rows), sheet);
  }
  const bytes = XLSX.write(book, { type: "array", bookType: "xlsx" });
  return new File([bytes], name);
}

const textOf = (file: File) => file.text();

describe("toCsvFile on a CSV", () => {
  it("returns the file untouched, with no sheets to choose from", async () => {
    const csv = new File(["smiles,y\nCCO,1\n"], "panel.csv", { type: "text/csv" });
    const converted = await toCsvFile(csv);
    expect(converted.file).toBe(csv);
    expect(converted.sheetNames).toEqual([]);
    expect(converted.sheet).toBeNull();
  });
});

describe("toCsvFile cell policy", () => {
  it("keeps a number's stored precision rather than its displayed rounding", async () => {
    // The cell displays "7.45" at two decimal places; the measurement is not 7.45.
    const sheet = XLSX.utils.aoa_to_sheet([
      ["smiles", "pIC50"],
      ["CCO", 7.4499998],
    ]);
    sheet.B2.z = "0.00";
    const book = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(book, sheet, "Sheet1");
    const bytes = XLSX.write(book, { type: "array", bookType: "xlsx" });
    const converted = await toCsvFile(new File([bytes], "potency.xlsx"));
    expect(await textOf(converted.file)).toBe("smiles,pIC50\nCCO,7.4499998\n");
  });

  it("writes a date as an ISO date, never an Excel serial number", async () => {
    // What a real .xlsx holds: the serial 46305 under a date number format.
    // Built by hand rather than from a JS Date, because SheetJS's writer shifts
    // a Date by the local zone and this suite runs pinned to America/Chicago.
    const sheet = XLSX.utils.aoa_to_sheet([
      ["compound", "measured_on"],
      ["CCO", 46305],
    ]);
    sheet.B2.z = "yyyy-mm-dd";
    const book = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(book, sheet, "Sheet1");
    const bytes = XLSX.write(book, { type: "array", bookType: "xlsx" });
    const converted = await toCsvFile(new File([bytes], "assay.xlsx"));
    expect(await textOf(converted.file)).toBe("compound,measured_on\nCCO,2026-10-10\n");
  });

  it("leaves a blank cell blank, so a sparse label is not read as a zero", async () => {
    const file = workbook({
      Sheet1: [
        ["smiles", "tox"],
        ["CCO", null],
        ["CCN", 1],
      ],
    });
    expect(await textOf((await toCsvFile(file)).file)).toBe("smiles,tox\nCCO,\nCCN,1\n");
  });

  it("passes a boolean through as TRUE or FALSE for validation to report", async () => {
    const file = workbook({
      Sheet1: [
        ["smiles", "active"],
        ["CCO", true],
        ["CCN", false],
      ],
    });
    expect(await textOf((await toCsvFile(file)).file)).toBe("smiles,active\nCCO,TRUE\nCCN,FALSE\n");
  });

  it("passes an error cell through as its text, for validation to reject by row", async () => {
    // Excel caches `#N/A` as an error cell. Blanking it would turn "the assay
    // failed here" into "not measured", which reads as a sparse label.
    const sheet = XLSX.utils.aoa_to_sheet([
      ["smiles", "y"],
      ["CCO", 0],
    ]);
    sheet.B2 = { t: "e", v: 0x2a, w: "#N/A" };
    const book = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(book, sheet, "Sheet1");
    const bytes = XLSX.write(book, { type: "array", bookType: "xlsx" });
    const converted = await toCsvFile(new File([bytes], "failed.xlsx"));
    expect(await textOf(converted.file)).toBe("smiles,y\nCCO,#N/A\n");
  });

  it("quotes a value containing a comma so the column count survives", async () => {
    const file = workbook({
      Sheet1: [
        ["smiles", "note"],
        ["CCO", 'a, b "c"'],
      ],
    });
    expect(await textOf((await toCsvFile(file)).file)).toBe('smiles,note\nCCO,"a, b ""c"""\n');
  });

  it("drops rows and columns outside the filled table", async () => {
    const file = workbook({
      Sheet1: [
        ["smiles", "y"],
        ["CCO", 1],
        [null, null],
        [null, null],
      ],
    });
    expect(await textOf((await toCsvFile(file)).file)).toBe("smiles,y\nCCO,1\n");
  });
});

describe("toCsvFile and sheets", () => {
  it("names the sheets and skips a prose sheet for the one holding the table", async () => {
    // What a supplementary workbook really looks like: the first sheet is a
    // line of prose rather than an empty sheet, so "the first sheet with any
    // data" picks the caption and imports one column of English.
    const file = workbook({
      Notes: [["Supplementary data for Figure 3."], ["Contact the authors."]],
      "Table S1": [
        ["smiles", "y"],
        ["CCO", 1],
        ["CCN", 2],
        ["CCC", 3],
      ],
      "Table S2": [
        ["smiles", "y"],
        ["CCO", 9],
      ],
    });
    const converted = await toCsvFile(file);
    expect(converted.sheetNames).toEqual(["Notes", "Table S1", "Table S2"]);
    expect(converted.sheet).toBe("Table S1");
    expect(await textOf(converted.file)).toBe("smiles,y\nCCO,1\nCCN,2\nCCC,3\n");
  });

  it("still takes a one-column sheet, which a prediction input legitimately is", async () => {
    const converted = await toCsvFile(workbook({ Only: [["smiles"], ["CCO"], ["CCN"]] }));
    expect(converted.sheet).toBe("Only");
    expect(await textOf(converted.file)).toBe("smiles\nCCO\nCCN\n");
  });

  it("converts the sheet asked for, so a picker can change it", async () => {
    const file = workbook({
      "Table S1": [
        ["smiles", "y"],
        ["CCO", 1],
      ],
      "Table S2": [
        ["smiles", "y"],
        ["CCN", 2],
      ],
    });
    const converted = await toCsvFile(file, "Table S2");
    expect(converted.sheet).toBe("Table S2");
    expect(await textOf(converted.file)).toBe("smiles,y\nCCN,2\n");
  });

  it("names the converted file after the workbook, as a CSV", async () => {
    const converted = await toCsvFile(
      workbook({ S: [["smiles"], ["CCO"]] }, "Supplementary 3.xlsx"),
    );
    expect(converted.file.name).toBe("Supplementary 3.csv");
  });

  it("refuses a workbook with no data in any sheet", async () => {
    await expect(toCsvFile(workbook({ Notes: [], Empty: [] }))).rejects.toThrow("no data");
  });
});
