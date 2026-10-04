import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SweepDetail } from "./sweep-detail";

function Wrapper({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () => new QueryClient({ defaultOptions: { queries: { retry: false } } }),
  );
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

const hoisted = vi.hoisted(() => ({
  calls: [] as { url: string; method: string; data?: unknown }[],
  gets: [] as string[],
}));

// The real retry hook runs, so what is asserted is the request it sends.
vi.mock("@/shared/lib/api/custom-instance", async () => {
  const actual = await vi.importActual<typeof import("@/shared/lib/api/custom-instance")>(
    "@/shared/lib/api/custom-instance",
  );
  return {
    ...actual,
    customInstance: async (config: { url: string; method: string; data?: unknown }) => {
      if (config.method === "GET") {
        hoisted.gets.push(config.url);
        return SWEEP;
      }
      hoisted.calls.push(config);
    },
  };
});

function member(id: string, status: string) {
  return {
    id,
    sweep_id: "sweep-1",
    name: `Config ${id}`,
    engine_id: "ecfp4-rf",
    conditions: {},
    status,
    progress: 1,
    phase: null,
    error_message: null,
    protocol_id: null,
    metrics: null,
    created_at: "2026-10-03T10:00:00Z",
  };
}

const SWEEP = {
  id: "sweep-1",
  name: "Solubility sweep",
  runs: [member("run-failed", "failed"), member("run-ready", "ready")],
};

describe("SweepDetail Resume", () => {
  beforeEach(() => {
    hoisted.calls.length = 0;
    hoisted.gets.length = 0;
  });

  it("shows Resume on a failed row only, and posts to that run's retry route", async () => {
    render(<SweepDetail id="sweep-1" />, { wrapper: Wrapper });

    const resume = await screen.findByRole("button", { name: "Resume" });
    expect(screen.getAllByRole("button", { name: "Resume" })).toHaveLength(1);

    fireEvent.click(resume);

    await waitFor(() => expect(hoisted.calls).toHaveLength(1));
    expect(hoisted.calls[0]).toEqual({ url: "/api/v1/runs/run-failed/retry", method: "POST" });
  });

  it("refetches the sweep after a row's Resume, because polling has stopped", async () => {
    render(<SweepDetail id="sweep-1" />, { wrapper: Wrapper });
    const resume = await screen.findByRole("button", { name: "Resume" });
    // Every member is terminal, so nothing polls: the next GET can only be the
    // invalidation, and without it the row would stay "Failed".
    expect(hoisted.gets).toEqual(["/api/v1/sweeps/sweep-1"]);

    fireEvent.click(resume);

    await waitFor(() => expect(hoisted.gets).toHaveLength(2));
    expect(hoisted.gets[1]).toBe("/api/v1/sweeps/sweep-1");
  });
});
