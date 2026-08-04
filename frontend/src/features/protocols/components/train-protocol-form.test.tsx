import { describe, expect, it } from "vitest";
import { comparesAgainstItself, resolveConditions } from "./train-protocol-form";

const SPECS = [{ key: "n_estimators", label: "Trees", type: "integer", default: 500, options: [] }];

describe("comparesAgainstItself", () => {
  it("is true when both sides fall back to the same defaults", () => {
    expect(comparesAgainstItself("rf", {}, "rf", {}, SPECS, SPECS)).toBe(true);
  });

  it("is true when one side sets a value the other takes as its default", () => {
    // The trap: comparing raw form state would call these different, and the
    // warning would go missing on a run the server treats as a self-comparison.
    expect(comparesAgainstItself("rf", { n_estimators: 500 }, "rf", {}, SPECS, SPECS)).toBe(true);
  });

  it("is false for the same engine with genuinely different settings", () => {
    expect(comparesAgainstItself("rf", { n_estimators: 100 }, "rf", {}, SPECS, SPECS)).toBe(false);
  });

  it("is false for different engines", () => {
    expect(comparesAgainstItself("rf", {}, "xgb", {}, SPECS, SPECS)).toBe(false);
  });
});
