import type { ScorecardResponse } from "@/shared/lib/api/model";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SplitDrawSpread } from "./scorecard-diagnostics";

const card = (over: Partial<ScorecardResponse>) =>
  ({
    primary_metric: "mcc",
    metrics: { mcc: 0.5 },
    split_strategy: "scaffold",
    replicate_summary: null,
    replicate_seeds: null,
    replicate_unavailable: null,
    ...over,
  }) as ScorecardResponse;

describe("the draw-spread panel", () => {
  it("renders nothing when no draws were requested", () => {
    // Both fields null. A user who did not ask for draws does not need it explained.
    const { container } = render(<SplitDrawSpread scorecard={card({})} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("explains a predefined split instead of showing a blank", () => {
    render(
      <SplitDrawSpread
        scorecard={card({
          replicate_unavailable:
            "Not applicable: the partitions come from a column in your file, so there is no second draw to take.",
        })}
      />,
    );
    expect(screen.getByText(/partitions come from a column in your file/i)).toBeInTheDocument();
  });

  it("shows the mean, the spread and how many draws there were", () => {
    render(
      <SplitDrawSpread
        scorecard={card({
          replicate_summary: { mcc: { mean: 0.48, sd: 0.03, n: 5 } },
          replicate_seeds: [2, 3, 4, 5, 6],
        })}
      />,
    );
    expect(screen.getByText(/0\.480/)).toBeInTheDocument();
    expect(screen.getByText(/0\.030/)).toBeInTheDocument();
    expect(screen.getByText(/5 draws/i)).toBeInTheDocument();
  });

  it("shows the table and the warning together when draws were lost", () => {
    // A spread over two draws when five were asked for is not the measurement that
    // was requested, and must not render as though it were.
    render(
      <SplitDrawSpread
        scorecard={card({
          replicate_summary: { mcc: { mean: 0.48, sd: 0.03, n: 2 } },
          replicate_seeds: [2, 3],
          replicate_unavailable:
            "2 of 5 split draws completed. The rest did not: the engine raised.",
        })}
      />,
    );
    expect(screen.getByText(/0\.480/)).toBeInTheDocument();
    expect(screen.getByText(/2 of 5 split draws completed/i)).toBeInTheDocument();
  });

  it("tells a metric undefined in every draw from one that was measured", () => {
    render(
      <SplitDrawSpread
        scorecard={card({
          metrics: { mcc: 0.5, auroc: 0.8 },
          replicate_summary: {
            mcc: { mean: 0.48, sd: 0.03, n: 3 },
            auroc: { mean: null, sd: null, n: 0 },
          },
          replicate_seeds: [2, 3, 4],
        })}
      />,
    );
    expect(screen.getByText(/undefined in every draw/i)).toBeInTheDocument();
    expect(screen.getByText(/3 draws/i)).toBeInTheDocument();
  });

  it("says which metric was undefined in only some of the draws", () => {
    // n below the number of completed draws is the metric's own story, and a
    // different one from a draw that never finished.
    render(
      <SplitDrawSpread
        scorecard={card({
          replicate_summary: { mcc: { mean: 0.48, sd: 0.03, n: 3 } },
          replicate_seeds: [2, 3, 4, 5, 6],
        })}
      />,
    );
    expect(screen.getByText(/3 of 5 draws/i)).toBeInTheDocument();
  });

  it("reports a mean without a spread from a single usable draw", () => {
    render(
      <SplitDrawSpread
        scorecard={card({
          replicate_summary: { mcc: { mean: 0.48, sd: null, n: 1 } },
          replicate_seeds: [2],
        })}
      />,
    );
    expect(screen.getByText(/0\.480/)).toBeInTheDocument();
    expect(screen.getByText(/one draw/i)).toBeInTheDocument();
  });

  it("says nothing was measured for a metric the summary omits", () => {
    render(
      <SplitDrawSpread
        scorecard={card({
          metrics: { mcc: 0.5, auroc: 0.8 },
          replicate_summary: { mcc: { mean: 0.48, sd: 0.03, n: 2 } },
          replicate_seeds: [2, 3],
        })}
      />,
    );
    expect(screen.getByText(/not measured/i)).toBeInTheDocument();
  });
});
