import { customInstance } from "@/shared/lib/api/custom-instance";
import type { RunResponse } from "@/shared/lib/api/model";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ProtocolList } from "./protocol-list";

vi.mock("@/shared/lib/api/custom-instance", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/shared/lib/api/custom-instance")>()),
  customInstance: vi.fn(),
}));

function Wrapper({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () => new QueryClient({ defaultOptions: { queries: { retry: false } } }),
  );
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

function trainingRun(overrides: Partial<RunResponse>): RunResponse {
  return {
    id: "run-1",
    workspace_id: "ws-1",
    kind: "training",
    status: "running",
    progress: 0.4,
    phase: "Fitting the engine",
    result_uri: null,
    error_message: null,
    protocol_id: null,
    metrics: null,
    lane: "cpu",
    name: "solubility model",
    created_at: "2026-10-03T12:00:00Z",
    ...overrides,
  };
}

/** The protocols list is empty; the runs list is whatever the test supplies. */
function serve(runs: RunResponse[]) {
  vi.mocked(customInstance).mockImplementation(async ({ url }) =>
    url.endsWith("/runs") ? { items: runs, next_cursor: null } : { items: [], next_cursor: null },
  );
}

describe("the In training section", () => {
  beforeEach(() => {
    vi.mocked(customInstance).mockReset();
  });

  it("lists a live run with its name and phase, linked to the run", async () => {
    serve([trainingRun({})]);
    render(<ProtocolList />, { wrapper: Wrapper });

    expect(await screen.findByText("In training")).toBeInTheDocument();
    const link = screen.getByRole("link", { name: /solubility model/ });
    expect(link).toHaveAttribute("href", "/runs/run-1");
    expect(link).toHaveTextContent("Fitting the engine");
    expect(screen.getByRole("progressbar")).toBeInTheDocument();
  });

  it("falls back to Training run and Queued for a run with neither yet", async () => {
    serve([trainingRun({ status: "pending", phase: null, name: null, progress: 0 })]);
    render(<ProtocolList />, { wrapper: Wrapper });

    const link = await screen.findByRole("link", { name: /Training run/ });
    expect(link).toHaveTextContent("Queued");
  });

  it("leaves out runs that have finished", async () => {
    serve([
      trainingRun({ id: "run-live", name: "still going" }),
      trainingRun({ id: "run-done", status: "ready", name: "finished model" }),
      trainingRun({ id: "run-bad", status: "failed", name: "failed model" }),
      trainingRun({ id: "run-off", status: "cancelled", name: "canceled model" }),
    ]);
    render(<ProtocolList />, { wrapper: Wrapper });

    expect(await screen.findByText("still going")).toBeInTheDocument();
    expect(screen.queryByText("finished model")).not.toBeInTheDocument();
    expect(screen.queryByText("failed model")).not.toBeInTheDocument();
    expect(screen.queryByText("canceled model")).not.toBeInTheDocument();
  });

  it("renders no section when nothing is live", async () => {
    serve([trainingRun({ status: "ready" })]);
    render(<ProtocolList />, { wrapper: Wrapper });

    expect(await screen.findByText("No protocols yet")).toBeInTheDocument();
    expect(screen.queryByText("In training")).not.toBeInTheDocument();
  });
});
