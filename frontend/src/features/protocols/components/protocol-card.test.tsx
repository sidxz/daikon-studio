import { customInstance } from "@/shared/lib/api/custom-instance";
import type { Headline } from "@/shared/lib/headlines";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Protocol } from "../types";
import { ProtocolCard } from "./protocol-card";

vi.mock("@/shared/lib/api/custom-instance", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/shared/lib/api/custom-instance")>()),
  customInstance: vi.fn(),
}));

const readout = (name: string, type: "numeric" | "probability" | "class" = "numeric") => ({
  name,
  type,
  unit: null,
  direction: null,
  description: "",
  threshold: null,
});

function protocol(overrides: Partial<Protocol> = {}): Protocol {
  return {
    id: "p1",
    name: "hERG model",
    status: "published",
    engine_id: "xgb",
    readouts: [readout("pIC50")],
    folder_id: null,
    created_by: "user-1",
    created_at: "2026-10-03T12:00:00Z",
    ...overrides,
  } as Protocol;
}

const headline = (overrides: Partial<Headline> = {}): Headline => ({
  column: "pIC50",
  primary_metric: "mcc",
  value: 0.603,
  baseline_value: 0.588,
  ...overrides,
});

function renderCard(props: Partial<Parameters<typeof ProtocolCard>[0]> = {}) {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <ProtocolCard
        protocol={protocol()}
        scores={[headline()]}
        engineName="XGBoost"
        creator="Ada Lovelace"
        {...props}
      />
    </QueryClientProvider>,
  );
}

describe("ProtocolCard", () => {
  beforeEach(() => {
    vi.mocked(customInstance).mockReset();
    vi.mocked(customInstance).mockResolvedValue({ items: [], can_edit: true });
  });

  it("leads with the score against its baseline, for a higher-is-better metric", () => {
    renderCard();
    expect(screen.getByRole("link", { name: "hERG model" })).toHaveAttribute(
      "href",
      "/protocols/p1",
    );
    expect(screen.getByText("Predicts pIC50")).toBeInTheDocument();
    expect(screen.getByText("MCC")).toBeInTheDocument();
    expect(screen.getByText("0.603")).toBeInTheDocument();
    expect(screen.getByText("Baseline 0.588")).toBeInTheDocument();
    expect(screen.getByText("0.015 better")).toBeInTheDocument();
  });

  it("calls a lower RMSE better and a higher one worse", () => {
    const { unmount } = renderCard({
      scores: [headline({ primary_metric: "rmse", value: 0.5, baseline_value: 0.6 })],
    });
    expect(screen.getByText("RMSE")).toBeInTheDocument();
    expect(screen.getByText("0.100 better")).toBeInTheDocument();
    unmount();

    renderCard({ scores: [headline({ primary_metric: "rmse", value: 0.7, baseline_value: 0.6 })] });
    expect(screen.getByText("0.100 worse")).toBeInTheDocument();
  });

  it("says Same as baseline on a tie", () => {
    renderCard({ scores: [headline({ value: 0.5, baseline_value: 0.5 })] });
    expect(screen.getByText("Same as baseline")).toBeInTheDocument();
  });

  it("says No score when the metric is undefined", () => {
    renderCard({ scores: [headline({ value: null })] });
    expect(screen.getByText("No score")).toBeInTheDocument();
    expect(screen.queryByText(/Baseline/)).not.toBeInTheDocument();
  });

  it("leaves the score out when no training run was found", () => {
    renderCard({ scores: [] });
    expect(screen.queryByText("MCC")).not.toBeInTheDocument();
    expect(screen.queryByText("No score")).not.toBeInTheDocument();
  });

  it("names a classifier's target once, not its probability column", () => {
    renderCard({
      protocol: protocol({
        readouts: [
          readout("herg_blocker_probability", "probability"),
          readout("herg_blocker", "class"),
        ],
      }),
      scores: [headline({ column: "herg_blocker" })],
    });
    expect(screen.getByText("Predicts herg_blocker")).toBeInTheDocument();
    expect(screen.queryByText(/more/)).not.toBeInTheDocument();
    // One target: the metric needs no "on".
    expect(screen.getByText("MCC")).toBeInTheDocument();
  });

  it("counts the other targets and says which one the score is on", () => {
    renderCard({
      protocol: protocol({
        readouts: [
          readout("aggregator_probability", "probability"),
          readout("aggregator", "class"),
          readout("herg_blocker_probability", "probability"),
          readout("herg_blocker", "class"),
          readout("logS"),
        ],
      }),
      scores: [headline({ column: "aggregator" }), headline({ column: "herg_blocker" })],
    });
    const subtitle = screen.getByText("Predicts aggregator and 2 more");
    expect(subtitle).toHaveAttribute("title", "aggregator, herg_blocker, logS");
    expect(screen.getByText("MCC on aggregator")).toBeInTheDocument();
    expect(screen.queryByText(/more target/)).not.toBeInTheDocument();
  });

  it("puts the engine, the creator and the date in the footer", () => {
    renderCard();
    expect(screen.getByText("XGBoost")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Ada Lovelace" })).toHaveTextContent("AL");
    expect(screen.getByText(/Oct 3/)).toBeInTheDocument();
  });

  it("gives a published card no badge and a draft a dashed edge and a Draft label", () => {
    const { container, unmount } = renderCard();
    expect(screen.queryByText("Published")).not.toBeInTheDocument();
    expect(screen.queryByText("Draft")).not.toBeInTheDocument();
    expect(container.firstElementChild?.className).not.toContain("border-dashed");
    unmount();

    const draft = renderCard({ protocol: protocol({ status: "draft" }) });
    expect(screen.getByText("Draft")).toBeInTheDocument();
    expect(draft.container.firstElementChild?.className).toContain("border-dashed");
  });
});
