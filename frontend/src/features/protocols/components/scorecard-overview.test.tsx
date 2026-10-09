import type { ScorecardResponse } from "@/shared/lib/api/model";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ScorecardOverview } from "./scorecard-overview";

function card(overrides: Partial<ScorecardResponse>): ScorecardResponse {
  return {
    prediction_kind: "value",
    metrics: { mae: 1, r2: -0.3 },
    baseline_metrics: { mae: 2 },
    baseline_is_self: false,
    test_count: 9000,
    parity: [{ actual: 0, predicted: 1 }],
    regression_summary: { mean_signed_error: -0.2, absolute_error_p90: 1.5 },
    unit: "nM",
    ...overrides,
  } as ScorecardResponse;
}

describe("plain-language performance overview", () => {
  it("explains negative R², signed bias and uses the full test count", () => {
    render(<ScorecardOverview scorecard={card({})} />);
    expect(screen.getByText("9,000 compounds")).toBeInTheDocument();
    expect(screen.getByText(/negative R² means/)).toBeInTheDocument();
    expect(screen.getByText(/too low/)).toHaveTextContent("0.200nM too low");
    expect(screen.getByText("50.0% lower")).toBeInTheDocument();
  });

  it("distinguishes undefined precision from zero recall", () => {
    render(
      <ScorecardOverview
        scorecard={card({
          prediction_kind: "probability",
          classification_summary: {
            true_positive: 0,
            false_positive: 0,
            false_negative: 20,
            true_negative: 80,
            precision: null,
            recall: 0,
            cutoff_inclusive: true,
          },
        })}
      />,
    );
    expect(screen.getByText(/No compounds were predicted active/)).toBeInTheDocument();
    expect(screen.getByText("0.0")).toBeInTheDocument();
    expect(screen.queryByText(/No active compounds were in the test set/)).not.toBeInTheDocument();
  });

  it("does not infer zero counts from an unavailable summary", () => {
    render(
      <ScorecardOverview
        scorecard={card({ prediction_kind: "probability", classification_summary: null })}
      />,
    );
    expect(screen.getByText("Precision is not available for this result.")).toBeInTheDocument();
    expect(screen.queryByText(/No compounds were predicted active/)).not.toBeInTheDocument();
  });
});

describe("unmeasured compounds", () => {
  it("says how many compounds the target was measured on when some were not", () => {
    render(
      <ScorecardOverview
        scorecard={card({ test_count: 400, labelled_test_rows: 120 })}
      />,
    );
    expect(screen.getByText(/Measured for 120 of 400/)).toBeInTheDocument();
  });

  it("says nothing extra when every compound was measured", () => {
    render(
      <ScorecardOverview
        scorecard={card({ test_count: 400, labelled_test_rows: 400 })}
      />,
    );
    expect(screen.queryByText(/Measured for/)).not.toBeInTheDocument();
  });
});
