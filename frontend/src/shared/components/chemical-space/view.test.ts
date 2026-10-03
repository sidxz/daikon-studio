import { describe, expect, it } from "vitest";
import { parseColor } from "./color";
import { buildPickGrid, fitView, pan, pickNearest, toMap, toScreen, zoomAt } from "./view";

describe("view", () => {
  const fit = fitView(400, 300, 10);

  it("fits the unit square into the shorter side, centred", () => {
    expect(fit).toEqual({ cx: 0.5, cy: 0.5, scale: 280 });
    expect(toScreen(fit, 400, 300, 0.5, 0.5)).toEqual([200, 150]);
  });

  it("fits the data's own bounds when given, filling a wide canvas", () => {
    const wide = fitView(400, 300, 10, [0, 0.25, 1, 0.75]);
    expect(wide.scale).toBe(380);
    expect(wide.cx).toBeCloseTo(0.5);
    expect(wide.cy).toBeCloseTo(0.5);
  });

  it("falls back to the unit square for a single point", () => {
    expect(fitView(400, 300, 10, [0.3, 0.3, 0.3, 0.3])).toEqual(fitView(400, 300, 10));
  });

  it("puts map y up and screen y down", () => {
    const [, top] = toScreen(fit, 400, 300, 0.5, 1);
    expect(top).toBeLessThan(150);
  });

  it("keeps the point under the cursor fixed while zooming", () => {
    const before = toMap(fit, 400, 300, 320, 60);
    const zoomed = zoomAt(fit, 400, 300, 320, 60, 2.5, fit);
    const after = toMap(zoomed, 400, 300, 320, 60);
    expect(after[0]).toBeCloseTo(before[0]);
    expect(after[1]).toBeCloseTo(before[1]);
  });

  it("clamps zoom between half and 64 times the fit", () => {
    expect(zoomAt(fit, 400, 300, 200, 150, 1000, fit).scale).toBe(fit.scale * 64);
    expect(zoomAt(fit, 400, 300, 200, 150, 0.001, fit).scale).toBe(fit.scale * 0.5);
  });

  it("pans by screen pixels", () => {
    const moved = pan(fit, 28, 0);
    expect(moved.cx).toBeCloseTo(0.4);
  });
});

describe("pick grid", () => {
  const xs = [0.1, 0.5, 0.52, 0.9];
  const ys = [0.1, 0.5, 0.5, 0.9];
  const grid = buildPickGrid(xs, ys, 16);

  it("finds the nearest point within the radius", () => {
    expect(pickNearest(grid, 0.515, 0.5, 0.05)).toBe(2);
  });

  it("finds nothing outside the radius", () => {
    expect(pickNearest(grid, 0.3, 0.3, 0.05)).toBe(-1);
  });
});

describe("parseColor", () => {
  it("reads hex and rgb()", () => {
    expect(parseColor("#3b82f6")).toEqual([59 / 255, 130 / 255, 246 / 255, 1]);
    expect(parseColor("#fff", 0.5)).toEqual([1, 1, 1, 0.5]);
    expect(parseColor("rgb(15, 23, 42)")).toEqual([15 / 255, 23 / 255, 42 / 255, 1]);
  });
});
