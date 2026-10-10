import { describe, expect, it } from "vitest";
import {
  columnRole,
  draftFromUpload,
  replaceUpload,
  setColumnRole,
  toggleTarget,
  withColumns,
} from "./draft-from-upload";

const rows = [
  { smiles: "CCO", solubility: "1.2", reactive: "0", id: "A1" },
  { smiles: "CCN", solubility: "3.4", reactive: "1", id: "A2" },
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

describe("column roles and replacement", () => {
  it("moves a target to the identifier role without leaving conflicting mappings", () => {
    const draft = draftFromUpload(columns, rows, "panel.csv");
    const next = setColumnRole(draft, "solubility", "identifier", rows);
    expect(next.idColumn).toBe("solubility");
    expect(next.targets).toEqual([]);
    expect(columnRole(next, "id")).toBe("unused");
  });

  it("clears the required structure choice if that column is deliberately reassigned", () => {
    const draft = draftFromUpload(columns, rows, "panel.csv");
    expect(setColumnRole(draft, "smiles", "unused", rows).structureColumn).toBe("");
  });

  it("preserves target metadata on a replacement with the same columns", () => {
    const draft = draftFromUpload(columns, rows, "panel.csv");
    draft.targets[0] = { ...draft.targets[0], unit: "µM", direction: "low" };
    const next = replaceUpload(draft, columns, rows, new File(["csv"], "corrected.csv"));
    expect(next.name).toBe("panel");
    expect(next.targets).toEqual(draft.targets);
    expect(next.structureColumn).toBe(draft.structureColumn);
    expect(next.idColumn).toBe(draft.idColumn);
  });

  it("removes vanished targets instead of silently substituting another measurement", () => {
    const draft = draftFromUpload(columns, rows, "panel.csv");
    const next = replaceUpload(
      draft,
      ["smiles", "id", "unrelated"],
      [],
      new File(["csv"], "replacement.csv"),
    );
    expect(next.targets).toEqual([]);
  });
});

describe("the split-assignment role", () => {
  const withSplit = ["smiles", "solubility", "reactive", "id", "split"];
  const splitRows = rows.map((row, index) => ({
    ...row,
    split: index === 0 ? "train" : "test",
  }));

  it("reports the role of the designated column", () => {
    const draft = setColumnRole(
      draftFromUpload(withSplit, splitRows, "panel.csv"),
      "split",
      "split",
      splitRows,
    );
    expect(draft.splitColumn).toBe("split");
    expect(columnRole(draft, "split")).toBe("split");
  });

  it("is exclusive: a target that becomes the split column stops being a target", () => {
    const draft = draftFromUpload(withSplit, splitRows, "panel.csv");
    const next = setColumnRole(draft, "solubility", "split", splitRows);
    expect(next.splitColumn).toBe("solubility");
    expect(next.targets.map((target) => target.column)).not.toContain("solubility");
  });

  it("is exclusive the other way: reassigning the split column clears it", () => {
    const draft = setColumnRole(
      draftFromUpload(withSplit, splitRows, "panel.csv"),
      "split",
      "split",
      splitRows,
    );
    const next = setColumnRole(draft, "split", "identifier", splitRows);
    expect(next.idColumn).toBe("split");
    expect(next.splitColumn).toBeNull();
  });

  it("drops a split column the replacement upload no longer has", () => {
    const draft = setColumnRole(
      draftFromUpload(withSplit, splitRows, "panel.csv"),
      "split",
      "split",
      splitRows,
    );
    const next = replaceUpload(draft, columns, rows, new File([], "other.csv"));
    expect(next.splitColumn).toBeNull();
  });

  it("keeps a split column the replacement upload still has", () => {
    const draft = setColumnRole(
      draftFromUpload(withSplit, splitRows, "panel.csv"),
      "split",
      "split",
      splitRows,
    );
    const next = replaceUpload(draft, withSplit, splitRows, new File([], "again.csv"));
    expect(next.splitColumn).toBe("split");
  });

  it("never lets the split column also be the structure column", () => {
    const draft = setColumnRole(
      draftFromUpload(withSplit, splitRows, "panel.csv"),
      "split",
      "split",
      splitRows,
    );
    const next = withColumns(draft, { structureColumn: "split" });
    expect(next.splitColumn).toBeNull();
  });
});

describe("replaceUpload and a target's kind", () => {
  it("re-guesses the kind from the replacement file, keeping the unit and direction", () => {
    const draft = {
      ...draftFromUpload(columns, rows, "panel.csv"),
      targets: [
        { column: "solubility", kind: "numeric" as const, unit: "uM", direction: "low" as const },
      ],
    };
    // The same column is 0/1 in the replacement: its kind is derived from the
    // data, so it is re-derived, while unit and direction are the scientist's.
    const binaryRows = [
      { smiles: "CCO", solubility: "1" },
      { smiles: "CCN", solubility: "0" },
    ];
    const next = replaceUpload(
      draft,
      ["smiles", "solubility"],
      binaryRows,
      new File([], "new.csv"),
    );
    expect(next.targets[0]).toEqual({
      column: "solubility",
      kind: "binary",
      unit: "uM",
      direction: "low",
    });
  });
});
