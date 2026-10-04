import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { epoch } from "../lib/training-epochs.fixture";
import { TrainingProgress } from "./training-progress";

describe("TrainingProgress, live", () => {
  it("leads with the model training now and tables the finished ones", () => {
    const points = [
      epoch(1, { target: "aggregator" }),
      epoch(2, { target: "aggregator", epochs: 2 }),
      epoch(1, { target: "reactive" }),
      epoch(2, { target: "reactive" }),
      epoch(3, { target: "reactive" }),
    ];

    render(
      <TrainingProgress
        points={points}
        live
        phase="reactive: Training Chemprop D-MPNN on cuda:0"
        progress={0.42}
      />,
    );

    expect(screen.getByText("42% of the run")).toBeInTheDocument();
    expect(screen.getByText(/Training now:/)).toHaveTextContent("Training now: reactive");
    expect(screen.getByText("3 of 50")).toBeInTheDocument(); // the current model's epoch
    expect(screen.getByText("about 47 min")).toBeInTheDocument(); // a minute an epoch
    expect(screen.getByText("Finished models")).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "aggregator" })).toBeInTheDocument();
  });
});
