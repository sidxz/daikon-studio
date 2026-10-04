import { describe, expect, it } from "vitest";
import { appliesToTasks, withoutInapplicable } from "./conditions";

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
