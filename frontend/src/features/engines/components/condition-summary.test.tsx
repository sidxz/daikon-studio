import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Condition } from "../types";
import { ConditionSummary } from "./condition-summary";

const weighting: Condition = {
  key: "positive_weighting",
  label: "Positive-class weighting",
  type: "enum",
  required: false,
  default: "none",
  minimum: null,
  maximum: null,
  options: ["none", "balanced", "sqrt_balanced"],
  option_labels: ["None", "Balanced", "Square-root balanced"],
  help: null,
  tasks: ["binary_classification"],
};

describe("ConditionSummary", () => {
  it("names enum options and the default by their labels", () => {
    render(<ConditionSummary conditions={[weighting]} />);
    expect(screen.getByText("None · Balanced · Square-root balanced")).toBeInTheDocument();
    expect(screen.getByText("default None")).toBeInTheDocument();
  });

  it("falls back to the raw values when an engine gives no labels", () => {
    render(<ConditionSummary conditions={[{ ...weighting, option_labels: [] }]} />);
    expect(screen.getByText("none · balanced · sqrt_balanced")).toBeInTheDocument();
    expect(screen.getByText("default none")).toBeInTheDocument();
  });
});
