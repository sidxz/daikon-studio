import type { ReadoutResponse } from "@/shared/lib/api/model";
import { describe, expect, it } from "vitest";
import { cutoffFor, formatThousandths, positionIn } from "./cell-scale";

describe("positionIn", () => {
  it("places a value on the run's own range, clamped to it", () => {
    expect(positionIn(5, 0, 10)).toBe(0.5);
    expect(positionIn(-3, 0, 10)).toBe(0);
    expect(positionIn(12, 0, 10)).toBe(1);
  });

  it("has no position without a range, and fills the bar for a range of one value", () => {
    expect(positionIn(5, null, 10)).toBeNull();
    expect(positionIn(5, undefined, undefined)).toBeNull();
    expect(positionIn(5, 5, 5)).toBe(1);
  });
});

describe("formatThousandths", () => {
  it("reads three decimals, and <0.001 for almost none instead of scientific notation", () => {
    expect(formatThousandths(0.2272)).toBe("0.227");
    expect(formatThousandths(3.29e-5)).toBe("<0.001");
    expect(formatThousandths(0)).toBe("0.000");
  });
});

describe("cutoffFor", () => {
  const readouts = (threshold: number | null) =>
    [
      { name: "p_np_probability", type: "probability", threshold: null },
      { name: "p_np", type: "class", threshold },
    ] as unknown as ReadoutResponse[];

  it("uses the tuned threshold stored on the target's class readout", () => {
    expect(cutoffFor("p_np_probability", readouts(0.37))).toBe(0.37);
  });

  it("falls back to 0.5 when the cutoff was not tuned", () => {
    expect(cutoffFor("p_np_probability", readouts(null))).toBe(0.5);
  });
});
