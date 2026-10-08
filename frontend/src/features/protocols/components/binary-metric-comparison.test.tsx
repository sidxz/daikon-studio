import type { ScorecardResponse } from "@/shared/lib/api/model";
import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BinaryMetricComparison } from "./binary-metric-comparison";

function card(overrides: Partial<ScorecardResponse> = {}): ScorecardResponse {
  return {
    prediction_kind: "probability",
    primary_metric: "mcc",
    primary_metric_ci: [0.48, 0.72],
    metrics: { mcc: 0.6, auprc: 0.42 },
    baseline_metrics: { mcc: 0.4, auprc: 0.5 },
    baseline_is_self: false,
    cutoff: 0.27,
    baseline_cutoff: 0.41,
    classification_summary: {
      true_positive: 15,
      false_negative: 5,
      false_positive: 10,
      true_negative: 170,
      precision: 0.6,
      recall: 0.75,
      cutoff_inclusive: true,
    },
    // A sampled plot may contain only one class. It must not drive availability
    // or the active share when full-test class counts say otherwise.
    parity: [{ actual: 0, predicted: 0.1, similarity: null }],
    ...overrides,
  } as ScorecardResponse;
}

describe("binary metric comparisons", () => {
  it("compares the saved scores independently and explains full-test prevalence and cutoffs", () => {
    render(<BinaryMetricComparison scorecard={card()} />);
    const mcc = within(screen.getByRole("region", { name: "MCC comparison" }));
    const pr = within(screen.getByRole("region", { name: "PR AUC comparison" }));
    expect(mcc.getByText("0.600")).toBeInTheDocument();
    expect(mcc.getByText("0.400")).toBeInTheDocument();
    expect(mcc.getByText("+0.200")).toBeInTheDocument();
    expect(mcc.getByText(/MCC is higher/)).toBeInTheDocument();
    expect(mcc.getByText("Ahead of baseline")).toHaveAttribute("data-variant", "success");
    expect(mcc.getByText(/cutoff 0.27; the comparison uses 0.41/)).toBeInTheDocument();
    expect(mcc.getByText(/95% interval/)).toHaveTextContent("0.480 to 0.720");
    expect(pr.getByText("0.420")).toBeInTheDocument();
    expect(pr.getByText("0.500")).toBeInTheDocument();
    expect(pr.getByText("-0.080")).toBeInTheDocument();
    expect(pr.getByText(/PR AUC is lower/)).toBeInTheDocument();
    expect(pr.getByText("Behind baseline")).toHaveAttribute("data-variant", "warning");
    expect(pr.getByText(/Calculated as average precision/)).toHaveTextContent(
      "20 of 200 test compounds are active (10.0%)",
    );
    expect(pr.queryByText(/95% interval/)).not.toBeInTheDocument();
  });

  it("still compares PR AUC when MCC is unavailable", () => {
    render(
      <BinaryMetricComparison
        scorecard={card({
          metrics: { mcc: null, auprc: 0.7 },
          metrics_undefined: { mcc: "MCC could not be computed for this run." },
        })}
      />,
    );
    const mcc = within(screen.getByRole("region", { name: "MCC comparison" }));
    const pr = within(screen.getByRole("region", { name: "PR AUC comparison" }));
    expect(mcc.getByText("MCC could not be computed for this run.")).toBeInTheDocument();
    expect(mcc.queryByText(/95% interval/)).not.toBeInTheDocument();
    expect(mcc.getByText("Not comparable")).toHaveAttribute("data-variant", "outline");
    expect(pr.getByText("0.700")).toBeInTheDocument();
    expect(pr.getByText("+0.200")).toBeInTheDocument();
    expect(pr.getByText("Ahead of baseline")).toHaveAttribute("data-variant", "success");
  });

  it("distinguishes a real zero from a missing comparison score", () => {
    render(
      <BinaryMetricComparison
        scorecard={card({
          metrics: { mcc: 0, auprc: 0.1 },
          baseline_metrics: { mcc: 0, auprc: null },
        })}
      />,
    );
    const mcc = within(screen.getByRole("region", { name: "MCC comparison" }));
    const pr = within(screen.getByRole("region", { name: "PR AUC comparison" }));
    expect(mcc.getAllByText("0.000")).toHaveLength(3);
    expect(mcc.getByText("Same score as the comparison model.")).toBeInTheDocument();
    expect(mcc.getByText("Matches baseline")).toBeInTheDocument();
    expect(pr.getAllByText("N/A")).toHaveLength(2);
    expect(pr.getByText("PR AUC is not available for the comparison model.")).toBeInTheDocument();
  });

  it.each(["active", "inactive", "empty"])(
    "explains the %s-only test set and hides misleading scores",
    (kind) => {
      render(
        <BinaryMetricComparison
          scorecard={card({
            classification_summary: {
              true_positive: kind === "active" ? 12 : 0,
              false_negative: 0,
              false_positive: 0,
              true_negative: kind === "inactive" ? 12 : 0,
              precision: null,
              recall: null,
              cutoff_inclusive: true,
            },
          })}
        />,
      );
      expect(
        screen.getByText(
          kind === "empty"
            ? /No test compounds are available/
            : new RegExp(`contains only ${kind} compounds`),
        ),
      ).toBeInTheDocument();
      expect(screen.queryByText("0.600")).not.toBeInTheDocument();
      expect(screen.queryByText("0.420")).not.toBeInTheDocument();
      expect(screen.queryByText(/95% interval/)).not.toBeInTheDocument();
      expect(screen.getAllByText("N/A")).toHaveLength(6);
    },
  );

  it("shows model scores without inventing a separate comparison when the baseline is itself", () => {
    render(<BinaryMetricComparison scorecard={card({ baseline_is_self: true })} />);
    expect(screen.getByText("0.600")).toBeInTheDocument();
    expect(screen.getByText("0.420")).toBeInTheDocument();
    expect(screen.queryByText("Comparison")).not.toBeInTheDocument();
    expect(screen.queryByText("Difference")).not.toBeInTheDocument();
    expect(screen.queryByText("0.400")).not.toBeInTheDocument();
    expect(screen.getAllByText("This is the baseline")).toHaveLength(2);
  });

  it("preserves the uncertain verdict only for the metric with a saved interval", () => {
    render(
      <BinaryMetricComparison scorecard={card({ baseline_metrics: { mcc: 0.55, auprc: 0.4 } })} />,
    );
    const mcc = within(screen.getByRole("region", { name: "MCC comparison" }));
    const pr = within(screen.getByRole("region", { name: "PR AUC comparison" }));
    expect(mcc.getByText("Ahead, but uncertain")).toHaveAttribute("data-variant", "warning");
    expect(mcc.getByText(/observed lead is uncertain/)).toBeInTheDocument();
    expect(pr.getByText("Ahead of baseline")).toHaveAttribute("data-variant", "success");
    expect(pr.queryByText(/observed lead is uncertain/)).not.toBeInTheDocument();
  });

  it("does not infer class counts from a sampled scatter when the summary is missing", () => {
    render(<BinaryMetricComparison scorecard={card({ classification_summary: null })} />);
    expect(screen.getByText("0.420")).toBeInTheDocument();
    expect(screen.getByText(/share of active test compounds is unavailable/)).toBeInTheDocument();
    expect(screen.queryByText(/contains only inactive/)).not.toBeInTheDocument();
  });
});

describe("a paired interval", () => {
  it("decides the MCC card and states the interval for the difference in place of the range", () => {
    render(
      <BinaryMetricComparison
        scorecard={card({ primary_metric_ci: [0.5, 0.7], difference_ci: [-0.01, 0.35] })}
      />,
    );
    const mcc = within(screen.getByRole("region", { name: "MCC comparison" }));
    expect(mcc.getByText("Ahead, but uncertain")).toBeInTheDocument();
    expect(mcc.getByText(/95% interval for the difference/)).toHaveTextContent("-0.010 to +0.350");
    expect(mcc.queryByText(/Likely range/)).not.toBeInTheDocument();
    expect(mcc.getByText(/The interval for the difference includes zero/)).toBeInTheDocument();
  });

  it("never lends the MCC difference to another metric's card", () => {
    render(
      <BinaryMetricComparison
        scorecard={card({
          metrics: { mcc: 0.6, auprc: 0.6 },
          baseline_metrics: { mcc: 0.4, auprc: 0.5 },
          difference_ci: [-0.01, 0.35],
        })}
      />,
    );
    const pr = within(screen.getByRole("region", { name: "PR AUC comparison" }));
    expect(pr.getByText("Ahead of baseline")).toBeInTheDocument();
    expect(pr.queryByText(/interval for the difference/)).not.toBeInTheDocument();
  });
});
