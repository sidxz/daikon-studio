import type { ReadoutResponse } from "@/shared/lib/api/model";
import { describe, expect, it } from "vitest";
import { targetsOf, uncertaintyColumn } from "./targets";

const readout = (name: string, type: ReadoutResponse["type"]): ReadoutResponse => ({
  name,
  type,
  unit: null,
  direction: null,
  description: "",
  threshold: null,
});

describe("targetsOf", () => {
  it("recovers each target once, in order, from its readouts", () => {
    expect(
      targetsOf([
        readout("solubility", "numeric"),
        readout("reactive_probability", "probability"),
        readout("reactive", "class"),
      ]),
    ).toEqual(["solubility", "reactive"]);
  });
});

describe("uncertaintyColumn", () => {
  it("keeps the plain name for one target and suffixes it for several", () => {
    expect(uncertaintyColumn("y", 1)).toBe("uncertainty");
    expect(uncertaintyColumn("y", 2)).toBe("y_uncertainty");
  });
});
