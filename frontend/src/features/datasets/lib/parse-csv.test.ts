import { describe, expect, it } from "vitest";
import { looksBinary } from "./parse-csv";

const column = (...values: string[]) => values.map((y) => ({ y }));

describe("looksBinary", () => {
  it("reads an all-zero column as binary, since a rare label may show one value", () => {
    expect(looksBinary(column("0", "0", "0"), "y")).toBe(true);
  });

  it("reads 0 and 1 as binary, including their numeric forms", () => {
    expect(looksBinary(column("0", "1", "1"), "y")).toBe(true);
    expect(looksBinary(column("1.0", "0.0"), "y")).toBe(true);
  });

  it("ignores empty cells", () => {
    expect(looksBinary(column("0", "", " ", "1"), "y")).toBe(true);
  });

  it("does not read two distinct numbers other than 0 and 1 as binary", () => {
    expect(looksBinary(column("2", "5"), "y")).toBe(false);
  });

  it("does not read text labels as binary", () => {
    expect(looksBinary(column("active", "inactive"), "y")).toBe(false);
  });

  it("does not read a measured column as binary", () => {
    expect(looksBinary(column("1.2", "3.4"), "y")).toBe(false);
    expect(looksBinary(column("0", "0.5", "1"), "y")).toBe(false);
  });

  it("does not read an empty column as binary", () => {
    expect(looksBinary(column("", " "), "y")).toBe(false);
    expect(looksBinary([], "y")).toBe(false);
  });
});

describe("the sample the guesses are made from", () => {
  it("reaches a sparse column's first measured value far down the file", async () => {
    const { parseCsvPreview } = await import("./parse-csv");
    // A real multi-task file: the 0/1 endpoint is blank for the first 200 rows.
    // Sampling too few rows guesses "numeric" having seen no value at all.
    const file = new File([`smiles,tox\n${"CCO,\n".repeat(200)}CCO,1\nCCO,0\n`], "sparse.csv");
    const preview = await parseCsvPreview(file);
    expect(looksBinary(preview.rows, "tox")).toBe(true);
  });
});

describe("CSV preview errors", () => {
  it("rejects a header-only file before setup begins", async () => {
    const { parseCsvPreview } = await import("./parse-csv");
    await expect(parseCsvPreview(new File(["smiles,y\n"], "empty.csv"))).rejects.toThrow(
      "no data rows",
    );
  });

  it("rejects malformed rows with a useful parsing reason", async () => {
    const { parseCsvPreview } = await import("./parse-csv");
    await expect(
      parseCsvPreview(new File(["smiles,y\nCCO,1,extra\n"], "broken.csv")),
    ).rejects.toThrow("Could not read this CSV");
  });
});
