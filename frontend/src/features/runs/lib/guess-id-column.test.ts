import { describe, expect, it } from "vitest";
import { guessIdColumn } from "./guess-id-column";

describe("guessIdColumn", () => {
  it("prefers an id-like header that is not the structure column", () => {
    expect(guessIdColumn(["Molecule Name", "smiles", "mw"], "smiles")).toBe("Molecule Name");
    expect(guessIdColumn(["compound_id", "smiles"], "smiles")).toBe("compound_id");
  });
  it("returns null when nothing looks like an identifier", () => {
    expect(guessIdColumn(["smiles", "mw"], "smiles")).toBe(null);
  });
});
