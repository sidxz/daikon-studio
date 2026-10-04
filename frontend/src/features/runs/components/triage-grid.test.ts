import { describe, expect, it } from "vitest";
import { columnHidden } from "./triage-grid";

describe("columnHidden", () => {
  it("hides SMILES until asked for, and shows everything else", () => {
    expect(columnHidden("smiles", {}, true)).toBe(true);
    expect(columnHidden("structure", {}, true)).toBe(false);
    expect(columnHidden("applicability", {}, true)).toBe(false);
  });
  it("follows this browser's choice either way", () => {
    expect(columnHidden("smiles", { smiles: true }, true)).toBe(false);
    expect(columnHidden("applicability", { applicability: false }, true)).toBe(true);
  });
  it("keeps the ID column hidden when the upload had no identifiers", () => {
    expect(columnHidden("compound_id", { compound_id: true }, false)).toBe(true);
    expect(columnHidden("compound_id", {}, true)).toBe(false);
  });
});
