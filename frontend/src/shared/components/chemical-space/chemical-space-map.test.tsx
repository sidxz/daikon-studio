import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ChemicalSpaceMap } from "./chemical-space-map";

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
