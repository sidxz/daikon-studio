import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RunDetail } from "./run-detail";

function Wrapper({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () => new QueryClient({ defaultOptions: { queries: { retry: false } } }),
  );
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

const hoisted = vi.hoisted(() => ({
  run: { current: {} as Record<string, unknown> },
  calls: [] as { url: string; method: string; data?: unknown }[],
}));

// The real retry hook runs, so what is asserted is the request it sends.
vi.mock("@/shared/lib/api/custom-instance", async () => {
  const actual = await vi.importActual<typeof import("@/shared/lib/api/custom-instance")>(
    "@/shared/lib/api/custom-instance",
  );
  return {
    ...actual,
    customInstance: async (config: { url: string; method: string; data?: unknown }) => {
      if (config.method === "GET") return hoisted.run.current;
      hoisted.calls.push(config);
    },
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));
vi.mock("@/features/collections", () => ({ useCreateCollection: () => ({ isPending: false }) }));
vi.mock("@/features/protocols", () => ({ useProtocol: () => ({ data: undefined }) }));
vi.mock("@/features/runners", () => ({ LANE_LABELS: {}, useRunners: () => ({ data: [] }) }));
vi.mock("./run-chemical-space", () => ({ RunChemicalSpace: () => null }));
vi.mock("./triage-grid", () => ({ TriageGrid: () => null }));

function stoppedRun(kind: string, status = "failed") {
  return {
    id: "run-1",
    kind,
    status,
    progress: 0.4,
    phase: null,
    error_message: "boom",
    protocol_id: null,
    metrics: null,
    lane: null,
    name: null,
    created_at: "2026-10-03T10:00:00Z",
  };
}

describe("RunDetail retry buttons", () => {
  beforeEach(() => {
    hoisted.calls.length = 0;
  });

  it("offers Resume and Start over on a failed training run", async () => {
    hoisted.run.current = stoppedRun("training");
    render(<RunDetail runId="run-1" />, { wrapper: Wrapper });

    expect(await screen.findByRole("button", { name: "Resume" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start over" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
  });

  it("Resume posts no body", async () => {
    hoisted.run.current = stoppedRun("training", "cancelled");
    render(<RunDetail runId="run-1" />, { wrapper: Wrapper });

    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));

    await waitFor(() => expect(hoisted.calls).toHaveLength(1));
    expect(hoisted.calls[0]).toEqual({ url: "/api/v1/runs/run-1/retry", method: "POST" });
  });

  it("Start over asks first, and posts fresh: true only once confirmed", async () => {
    hoisted.run.current = stoppedRun("training");
    render(<RunDetail runId="run-1" />, { wrapper: Wrapper });

    fireEvent.click(await screen.findByRole("button", { name: "Start over" }));

    const dialog = await screen.findByRole("alertdialog");
    expect(within(dialog).getByText("Start this run over?")).toBeInTheDocument();
    expect(
      within(dialog).getByText(
        "Its saved progress is discarded and training begins again from the start.",
      ),
    ).toBeInTheDocument();
    expect(hoisted.calls).toHaveLength(0);

    fireEvent.click(within(dialog).getByRole("button", { name: "Start over" }));

    await waitFor(() => expect(hoisted.calls).toHaveLength(1));
    expect(hoisted.calls[0]).toEqual({
      url: "/api/v1/runs/run-1/retry",
      method: "POST",
      data: { fresh: true },
    });
  });

  it("Cancel in the Start over dialog sends nothing", async () => {
    hoisted.run.current = stoppedRun("training");
    render(<RunDetail runId="run-1" />, { wrapper: Wrapper });

    fireEvent.click(await screen.findByRole("button", { name: "Start over" }));
    const dialog = await screen.findByRole("alertdialog");
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));

    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull());
    expect(hoisted.calls).toHaveLength(0);
  });

  it("keeps Retry, and only Retry, on a failed prediction run", async () => {
    hoisted.run.current = stoppedRun("prediction");
    render(<RunDetail runId="run-1" />, { wrapper: Wrapper });

    fireEvent.click(await screen.findByRole("button", { name: "Retry" }));

    expect(screen.queryByRole("button", { name: "Resume" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Start over" })).toBeNull();
    await waitFor(() => expect(hoisted.calls).toHaveLength(1));
    expect(hoisted.calls[0]).toEqual({ url: "/api/v1/runs/run-1/retry", method: "POST" });
  });
});
