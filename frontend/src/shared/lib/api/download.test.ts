import { describe, expect, it } from "vitest";
import { fileName } from "./download";

describe("fileName", () => {
  it("names a file after what the scientist called it, one dash per gap", () => {
    expect(fileName("Nuisance panel - XGBoost (UI test) predictions 2026-10-04", "xlsx", "x")).toBe(
      "nuisance-panel-xgboost-ui-test-predictions-2026-10-04.xlsx",
    );
    expect(fileName("  !!  ", "csv", "collection")).toBe("collection.csv");
  });
});
