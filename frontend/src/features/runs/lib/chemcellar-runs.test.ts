import { describe, expect, it } from "vitest";
import { activeCompounds, formatRunDate, runOption } from "./chemcellar-runs";

const run = {
  id: "r",
  protocol_id: "p",
  run_date: "2026-06-05",
  status: "draft",
  measured_count: 60,
  plate_count: 1,
  plate_barcodes: ["P1"],
};

describe("formatRunDate", () => {
  it("is the calendar date ChemCellar recorded, in any time zone", () => {
    expect(formatRunDate("2026-06-05")).toBe("Jun 5, 2026");
  });
});

describe("runOption", () => {
  it("reads as date, status, measured compounds and plates", () => {
    expect(runOption(run)).toEqual({
      label: "Jun 5, 2026 · Draft · 60 compounds measured · 1 plate",
      disabled: false,
    });
  });
  it("is selectable with plates but no readouts yet", () => {
    expect(runOption({ ...run, measured_count: 0 }).disabled).toBe(false);
  });
  it("is disabled with neither readouts nor plates", () => {
    expect(runOption({ ...run, measured_count: 0, plate_count: 0 })).toEqual({
      label: "Jun 5, 2026 · Draft · No compounds",
      disabled: true,
    });
  });
});

describe("activeCompounds", () => {
  const imported = { compound_count: 60 } as never;
  it("counts only the tab in view", () => {
    expect(activeCompounds("csv", 12, imported)).toEqual({ count: 12, ready: true });
    expect(activeCompounds("chemcellar", 12, imported)).toEqual({ count: 60, ready: true });
    expect(activeCompounds("chemcellar", 12, null)).toEqual({ count: 0, ready: false });
    expect(activeCompounds("csv", 0, imported)).toEqual({ count: 0, ready: false });
  });
});
