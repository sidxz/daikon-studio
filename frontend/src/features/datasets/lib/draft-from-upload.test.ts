import { describe, expect, it } from "vitest";
import { draftFromUpload, toggleTarget, withColumns } from "./draft-from-upload";

const rows = [
  { smiles: "CCO", solubility: "1.2", reactive: "0", id: "A1" },
  { smiles: "CCN", solubility: "3.4", reactive: "1", id: "A2" },
  // A third distinct value, or the two-valued solubility column reads as binary.
  { smiles: "CCC", solubility: "5.6", reactive: "1", id: "A3" },
];
const columns = ["smiles", "solubility", "reactive", "id"];

describe("draftFromUpload", () => {
  it("starts with the first non-structure column as the only target", () => {
    const draft = draftFromUpload(columns, rows, "panel.csv");
    expect(draft.targets).toEqual([
      { column: "solubility", kind: "numeric", unit: "", direction: "high" },
    ]);
  });

  it("guesses structure, target and identifier columns from the headers", () => {
    const draft = draftFromUpload(["Structure", "pIC50", "NATHAN ID"], [], "screen.csv");
    expect(draft).toMatchObject({
      name: "screen",
      structureColumn: "Structure",
      idColumn: "NATHAN ID",
    });
    expect(draft.targets.map((target) => target.column)).toEqual(["pIC50"]);
  });

  it("leaves a leading identifier column out of the targets", () => {
    const draft = draftFromUpload(["Compound_ID", "SMILES", "pIC50"], [], "screen.csv");
    expect(draft.targets.map((target) => target.column)).toEqual(["pIC50"]);
    expect(draft.idColumn).toBe("Compound_ID");
  });

  it("pre-checks nothing when only the structure and the identifier remain", () => {
    const draft = draftFromUpload(["smiles", "name"], [], "a.csv");
    expect(draft.targets).toEqual([]);
    expect(draft.idColumn).toBe("name");
  });

  it("never guesses the target column as the identifier", () => {
    const draft = draftFromUpload(["smiles", "y", "compound_id"], [], "a.csv");
    expect(draft.targets.map((target) => target.column)).toEqual(["y"]);
    expect(draft.idColumn).toBe("compound_id");
  });
});

describe("toggleTarget", () => {
  it("adds targets in the order chosen, guessing each one's kind, and removes them", () => {
    const draft = draftFromUpload(columns, rows, "panel.csv");
    const both = toggleTarget(draft, "reactive", true, rows);
    expect(both.targets.map((t) => [t.column, t.kind])).toEqual([
      ["solubility", "numeric"],
      ["reactive", "binary"],
    ]);
    expect(toggleTarget(both, "solubility", false, rows).targets.map((t) => t.column)).toEqual([
      "reactive",
    ]);
  });

  it("clears an identifier that becomes a target", () => {
    const draft = { ...draftFromUpload(columns, rows, "panel.csv"), idColumn: "id" };
    expect(toggleTarget(draft, "id", true, rows).idColumn).toBeNull();
  });
});

describe("withColumns", () => {
  it("drops a target that becomes the structure column", () => {
    const draft = draftFromUpload(columns, rows, "panel.csv");
    expect(withColumns(draft, { structureColumn: "solubility" }).targets).toEqual([]);
  });

  it("clears the identifier when it becomes the structure column", () => {
    const draft = draftFromUpload(["smiles", "y", "name"], [], "a.csv");
    expect(withColumns(draft, { structureColumn: "name" }).idColumn).toBeNull();
  });
});
