import { fireEvent, render, screen } from "@testing-library/react";
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
  option_labels: [] as string[],
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
  option_labels: [] as string[],
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

describe("ConditionFields option labels", () => {
  const LABELED = { ...WEIGHTING, option_labels: ["None", "Balanced"] };

  it("shows each option's label and submits the option itself", async () => {
    const onChange = vi.fn();
    render(<ConditionFields conditions={[LABELED]} values={{}} onChange={onChange} />);

    fireEvent.click(screen.getByRole("combobox"));
    expect(await screen.findByRole("option", { name: "None" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "balanced" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("option", { name: "Balanced" }));

    expect(onChange).toHaveBeenCalledWith("positive_weighting", "balanced");
  });

  it("shows the raw options when the setting has no labels", async () => {
    render(<ConditionFields conditions={[WEIGHTING]} values={{}} onChange={vi.fn()} />);

    fireEvent.click(screen.getByRole("combobox"));
    expect(await screen.findByRole("option", { name: "balanced" })).toBeInTheDocument();
  });
});
