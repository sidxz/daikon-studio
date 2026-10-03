import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ChemicalSpaceMap, tooltipPlacement } from "./chemical-space-map";

describe("ChemicalSpaceMap", () => {
  it("falls back to a sentence when WebGL2 is unavailable", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    render(
      <ChemicalSpaceMap
        base={{ x: [0.1, 0.9], y: [0.2, 0.8], style: [0, 0] }}
        colors={[
          [0, 0, 1, 1],
          [0, 1, 1, 1],
          [1, 0.5, 0, 1],
          [1, 0.5, 0, 1],
          [1, 0.5, 0, 1],
          [0, 0, 0, 1],
        ]}
        label="2 training compounds"
      />,
    );
    expect(screen.getByText("This browser cannot draw the map.")).toBeInTheDocument();
  });
});

describe("tooltipPlacement", () => {
  it("opens below and right of the cursor in the top-left of the map", () => {
    expect(tooltipPlacement(100, 80, 1000, 420)).toMatchObject({
      left: 114,
      top: 94,
      transform: "translate(0, 0)",
    });
  });

  it("flips above and left near the bottom-right, so the last line is never cut off", () => {
    expect(tooltipPlacement(900, 400, 1000, 420)).toMatchObject({
      left: 886,
      top: 386,
      transform: "translate(-100%, -100%)",
    });
  });
});
