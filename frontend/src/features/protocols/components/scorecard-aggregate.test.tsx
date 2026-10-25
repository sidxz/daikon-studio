import type { ScorecardResponse } from "@/shared/lib/api/model";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ScorecardAggregate } from "./scorecard-aggregate";

/** The real aggregate is used throughout: it is pure, and a fake would prove nothing. */
const card = (target: string, value: number | null, rest: Partial<ScorecardResponse> = {}) =>
  ({
    target,
    joint_model: true,
    primary_metric: "auroc",
    metrics: { auroc: value },
    metrics_undefined: null,
    baseline_metrics: null,
    unit: null,
    ...rest,
  }) as ScorecardResponse;

const row = (name: string) => screen.getByRole("rowheader", { name }).closest("tr");

describe("ScorecardAggregate", () => {
  it("leads with the average, named by its metric and its target count", () => {
    render(<ScorecardAggregate scorecards={[card("a", 0.8), card("b", 0.9)]} />);

    expect(screen.getByText("Average ROC AUC across 2 targets")).toBeInTheDocument();
    expect(screen.getByText("0.850")).toBeInTheDocument();
  });

  it("says every target counts once, which is what makes it comparable", () => {
    render(<ScorecardAggregate scorecards={[card("a", 0.8), card("b", 0.9)]} />);

    expect(screen.getByText(/Every target counts once/)).toBeInTheDocument();
  });

  it("averages the comparison model too, and refuses to call the difference a test", () => {
    render(
      <ScorecardAggregate
        scorecards={[
          card("a", 0.8, { baseline_metrics: { auroc: 0.7 } }),
          card("b", 0.9, { baseline_metrics: { auroc: 0.75 } }),
        ]}
      />,
    );

    expect(screen.getByText("0.725")).toBeInTheDocument();
    expect(screen.getByText("+0.125")).toBeInTheDocument();
    expect(screen.getByText(/in this model's favor/)).toBeInTheDocument();
    expect(screen.getByText(/not a test of significance/)).toBeInTheDocument();
  });

  it("says plainly when the difference runs against the model", () => {
    render(
      <ScorecardAggregate
        scorecards={[
          card("a", 0.6, { baseline_metrics: { auroc: 0.8 } }),
          card("b", 0.6, { baseline_metrics: { auroc: 0.8 } }),
        ]}
      />,
    );

    expect(screen.getByText("-0.200")).toBeInTheDocument();
    expect(screen.getByText(/against it/)).toBeInTheDocument();
  });

  it("leaves out the comparison entirely when there is none", () => {
    render(<ScorecardAggregate scorecards={[card("a", 0.8), card("b", 0.9)]} />);

    expect(screen.queryByText("Comparison model")).not.toBeInTheDocument();
    expect(screen.queryByText("Difference")).not.toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "Comparison" })).not.toBeInTheDocument();
  });

  it("lists every target, weakest first", () => {
    render(
      <ScorecardAggregate
        scorecards={[card("strong", 0.95), card("weak", 0.55), card("middling", 0.75)]}
      />,
    );

    expect(screen.getAllByRole("rowheader").map((cell) => cell.textContent)).toEqual([
      "weak",
      "middling",
      "strong",
    ]);
    expect(row("weak")).toHaveTextContent("0.550");
  });

  it("names a target left out of the average, and what the average is over", () => {
    render(
      <ScorecardAggregate
        scorecards={[
          card("a", 0.8),
          card("b", 0.9),
          card("blank", null, {
            metrics_undefined: { auroc: "no compound in the test set carries a label" },
          }),
        ]}
      />,
    );

    expect(screen.getByText(/1 target left out of the average/)).toBeInTheDocument();
    expect(screen.getByText(/The average includes 2 of 3 targets/)).toBeInTheDocument();
    expect(screen.getByText(/no compound in the test set carries a label/)).toBeInTheDocument();
    expect(screen.queryByRole("rowheader", { name: "blank" })).not.toBeInTheDocument();
  });

  it("explains itself rather than averaging targets scored on different metrics", () => {
    render(
      <ScorecardAggregate
        scorecards={[
          card("auc", 0.8),
          card("err", 0.3, { primary_metric: "mae", metrics: { mae: 0.3 } }),
        ]}
      />,
    );

    expect(screen.getByText("No average across targets")).toBeInTheDocument();
    expect(screen.getByText(/different metrics/)).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("offers nothing to choose when the targets carry one metric", () => {
    render(<ScorecardAggregate scorecards={[card("a", 0.8), card("b", 0.9)]} />);

    expect(screen.queryByLabelText("Metric to average")).not.toBeInTheDocument();
  });

  it("offers the shared metrics, opening on the primary one", () => {
    const both = (target: string, mcc: number, auroc: number) =>
      card(target, null, { primary_metric: "mcc", metrics: { mcc, auroc } });
    render(<ScorecardAggregate scorecards={[both("a", 0.2, 0.8), both("b", 0.4, 0.9)]} />);

    expect(screen.getByLabelText("Metric to average")).toHaveTextContent("MCC");
    expect(screen.getByText("Average MCC across 2 targets")).toBeInTheDocument();
  });

  it("keeps the picker when the chosen metric cannot be averaged", () => {
    // Otherwise choosing an error metric whose targets disagree about units is
    // a dead end: a refusal with nothing on screen that would undo it.
    const mae = (target: string, unit: string) =>
      card(target, null, {
        primary_metric: "mae",
        metrics: { mae: 0.5, auroc: 0.8 },
        unit,
      });
    render(<ScorecardAggregate scorecards={[mae("a", "nM"), mae("b", "log M")]} />);

    expect(screen.getByText("No average across targets")).toBeInTheDocument();
    expect(screen.getByLabelText("Metric to average")).toBeInTheDocument();
  });

  it("carries the unit on an error metric, where the number has one", () => {
    const mae = (target: string, value: number) =>
      card(target, null, {
        primary_metric: "mae",
        metrics: { mae: value },
        unit: "log M",
      });
    render(<ScorecardAggregate scorecards={[mae("a", 0.4), mae("b", 0.6)]} />);

    expect(screen.getByText("Average MAE across 2 targets")).toBeInTheDocument();
    expect(screen.getAllByText("log M").length).toBeGreaterThan(0);
    expect(screen.getByText(/Lower is better/)).toBeInTheDocument();
  });
});
