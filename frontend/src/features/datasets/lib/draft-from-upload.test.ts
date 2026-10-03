import { describe, expect, it } from "vitest";
import { draftFromUpload, withColumns } from "./draft-from-upload";

describe("draftFromUpload", () => {
  it("guesses structure, target and identifier columns from the headers", () => {
    const draft = draftFromUpload(["Structure", "pIC50", "NATHAN ID"], [], "screen.csv");
    expect(draft).toMatchObject({
      name: "screen",
      structureColumn: "Structure",
      targetColumn: "pIC50",
      idColumn: "NATHAN ID",
    });
  });

  it("never guesses the target column as the identifier", () => {
    const draft = draftFromUpload(["smiles", "y", "compound_id"], [], "a.csv");
    expect(draft).toMatchObject({ targetColumn: "y", idColumn: "compound_id" });
  });
});

describe("withColumns", () => {
  it("clears the identifier when it becomes the structure or target column", () => {
    const draft = draftFromUpload(["smiles", "y", "name"], [], "a.csv");
    expect(withColumns(draft, { targetColumn: "name" }).idColumn).toBeNull();
    expect(withColumns(draft, { structureColumn: "name" }).idColumn).toBeNull();
    expect(withColumns(draft, { targetColumn: "y" }).idColumn).toBe("name");
  });
});
