import { describe, expect, it } from "vitest";
import { appliesToTasks, conditionError, conditionsValid, withoutInapplicable } from "./conditions";

describe("conditionError", () => {
  const ENSEMBLE = { type: "integer", minimum: 1, maximum: 10 };

  it("refuses a value outside the setting's bounds, in the form's words", () => {
    expect(conditionError(ENSEMBLE, 12)).toBe("Must be between 1 and 10.");
    expect(conditionError(ENSEMBLE, 0)).toBe("Must be between 1 and 10.");
    expect(conditionError({ type: "number", minimum: 0, maximum: null }, -1)).toBe(
      "Must be at least 0.",
    );
  });

  it("refuses a fraction for a whole-number setting", () => {
    expect(conditionError(ENSEMBLE, 2.5)).toBe("Enter a whole number.");
  });

  it("accepts an in-range value, an empty field (the default applies) and non-numbers", () => {
    expect(conditionError(ENSEMBLE, 3)).toBeNull();
    expect(conditionError(ENSEMBLE, undefined)).toBeNull();
    expect(conditionError({ type: "enum", minimum: null, maximum: null }, "balanced")).toBeNull();
  });
});

describe("conditionsValid", () => {
  const ENSEMBLE = { key: "ensemble_size", type: "integer", minimum: 1, maximum: 10, tasks: [] };

  it("is false while any shown setting is out of range", () => {
    expect(conditionsValid([ENSEMBLE], { ensemble_size: 12 }, undefined)).toBe(false);
    expect(conditionsValid([ENSEMBLE], { ensemble_size: 5 }, undefined)).toBe(true);
  });

  it("ignores a setting the pretrained weights fix", () => {
    expect(
      conditionsValid([ENSEMBLE], { ensemble_size: 12 }, undefined, { ensemble_size: 1 }),
    ).toBe(true);
  });
});

const WEIGHTING = { key: "positive_weighting", tasks: ["binary_classification"] };
const EPOCHS = { key: "epochs", tasks: [] };

describe("appliesToTasks", () => {
  it("needs a shared task when the setting names tasks", () => {
    expect(appliesToTasks(WEIGHTING, ["regression"])).toBe(false);
    expect(appliesToTasks(WEIGHTING, ["regression", "binary_classification"])).toBe(true);
  });

  it("always applies an unscoped setting, and everything before a dataset is chosen", () => {
    expect(appliesToTasks(EPOCHS, ["regression"])).toBe(true);
    expect(appliesToTasks(WEIGHTING, undefined)).toBe(true);
  });
});

describe("withoutInapplicable", () => {
  const values = { positive_weighting: "balanced", epochs: 5, other: 1 };

  it("drops only the keys whose settings do not apply to the dataset", () => {
    expect(withoutInapplicable([WEIGHTING, EPOCHS], values, ["regression"])).toEqual({
      epochs: 5,
      other: 1,
    });
  });

  it("keeps everything when the setting applies", () => {
    expect(withoutInapplicable([WEIGHTING, EPOCHS], values, ["binary_classification"])).toEqual(
      values,
    );
  });
});
