import { describe, expect, it } from "vitest";

import { parseTable, resolveColumns } from "@/features/pages/lib/parse-table";

import { CHART_PRESETS } from "./chart-presets";

const KINDS = ["bar", "line", "scatter", "radar", "pie"];

describe("CHART_PRESETS", () => {
  it("covers the eight domain charts", () => {
    expect(Object.keys(CHART_PRESETS).sort()).toEqual(
      [
        "composition",
        "doseResponse",
        "featureImportance",
        "kaplanMeier",
        "parity",
        "propertyProfile",
        "roc",
        "volcano",
      ].sort(),
    );
  });

  it("every preset names a real kind and carries a usable sample table", () => {
    for (const [id, p] of Object.entries(CHART_PRESETS)) {
      expect(KINDS, id).toContain(p.kind);
      expect(p.label.length, id).toBeGreaterThan(0);
      const table = parseTable(p.sample);
      // A sample the dialog shows must actually plot, or the preview lies.
      expect(table.rows.length, id).toBeGreaterThan(0);
      expect(resolveColumns(table, null, null).series.length, id).toBeGreaterThan(0);
    }
  });

  it("uses well-formed reference lines only", () => {
    for (const [id, p] of Object.entries(CHART_PRESETS)) {
      for (const r of p.options.refLines ?? []) {
        if ("segment" in r) {
          expect(["diagonal", "identity"], id).toContain(r.segment);
        } else {
          expect(["x", "y"], id).toContain(r.axis);
          expect(Number.isFinite(r.value), id).toBe(true);
        }
      }
    }
  });

  it("gives the ML and pharmacology presets the axes they are defined by", () => {
    expect(CHART_PRESETS.doseResponse.options.logX).toBe(true);
    expect(CHART_PRESETS.featureImportance.options.horizontal).toBe(true);
    expect(CHART_PRESETS.kaplanMeier.options.step).toBe(true);
    expect(CHART_PRESETS.roc.options.refLines).toEqual([{ segment: "diagonal" }]);
    expect(CHART_PRESETS.parity.options.refLines).toEqual([{ segment: "identity" }]);
    // ±1 log2 fold change and p = 0.05 as -log10.
    expect(CHART_PRESETS.volcano.options.refLines).toEqual([
      { axis: "x", value: -1 },
      { axis: "x", value: 1 },
      { axis: "y", value: 1.3, label: "p = 0.05" },
    ]);
  });
});
