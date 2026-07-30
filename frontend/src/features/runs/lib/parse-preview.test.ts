import { describe, expect, it } from "vitest";
import { PREVIEW_SAMPLE_SIZE, summarisePreview } from "./parse-preview";

const rows = [
  { smiles: "CCO", note: "a" },
  { smiles: "CCC", note: "b" },
  { smiles: "", note: "c" },
  { smiles: "  ", note: "d" },
];

describe("prediction upload preview", () => {
  it("counts every row in the file", () => {
    expect(summarisePreview(rows, "smiles").total).toBe(4);
  });

  it("counts blank cells separately: they are not structures", () => {
    expect(summarisePreview(rows, "smiles").blank).toBe(2);
  });

  it("samples the first few non-blank structures for rendering", () => {
    expect(summarisePreview(rows, "smiles").sample).toEqual(["CCO", "CCC"]);
  });

  it("caps the sample", () => {
    const many = Array.from({ length: 50 }, (_, index) => ({ smiles: `C${"C".repeat(index)}` }));
    expect(summarisePreview(many, "smiles").sample).toHaveLength(PREVIEW_SAMPLE_SIZE);
  });

  it("reports an empty summary for a column that is not there", () => {
    expect(summarisePreview(rows, "nope")).toEqual({ total: 4, sample: [], blank: 4 });
  });
});
