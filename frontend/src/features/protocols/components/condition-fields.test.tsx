import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ConditionFields } from "./condition-fields";

const WEIGHTING = {
  key: "positive_weighting",
  label: "Positive-class weighting",
  type: "enum",
  required: false,
  default: "none",
  minimum: null,
  maximum: null,
  options: ["none", "balanced"],
  help: null,
  tasks: ["binary_classification"],
};
const EPOCHS = {
  key: "epochs",
  label: "Training epochs",
  type: "integer",
  required: false,
  default: 50,
  minimum: 1,
  maximum: 500,
  options: [],
  help: null,
  tasks: [],
};

function show(tasks?: string[]) {
  render(
    <ConditionFields
      conditions={[WEIGHTING, EPOCHS]}
      values={{}}
      onChange={vi.fn()}
      tasks={tasks}
    />,
  );
}

describe("ConditionFields task scoping", () => {
  it("hides a setting that applies only to tasks the dataset does not have", () => {
    show(["regression"]);
    expect(screen.queryByText("Positive-class weighting")).not.toBeInTheDocument();
    expect(screen.getByText("Training epochs")).toBeInTheDocument();
  });

  it("shows a setting when the dataset has one of its tasks", () => {
    show(["regression", "binary_classification"]);
    expect(screen.getByText("Positive-class weighting")).toBeInTheDocument();
  });

  it("shows every setting while the dataset's tasks are unknown", () => {
    show(undefined);
    expect(screen.getByText("Positive-class weighting")).toBeInTheDocument();
    expect(screen.getByText("Training epochs")).toBeInTheDocument();
  });
});
