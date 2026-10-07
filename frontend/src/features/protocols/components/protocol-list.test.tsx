import { customInstance } from "@/shared/lib/api/custom-instance";
import type { RunResponse } from "@/shared/lib/api/model";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ProtocolList } from "./protocol-list";

vi.mock("@/shared/lib/api/custom-instance", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/shared/lib/api/custom-instance")>()),
  customInstance: vi.fn(),
}));

const nav = vi.hoisted(() => ({ replace: vi.fn(), query: "" }));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: nav.replace }),
  usePathname: () => "/protocols",
  useSearchParams: () => new URLSearchParams(nav.query),
}));
vi.mock("@/shared/lib/auth/use-workspace-members", () => ({
  useMemberName: () => (id: string | null | undefined) => (id === "user-1" ? "Ada" : undefined),
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
    requested_by: "user-1",
    source: null,
    created_at: "2026-10-03T12:00:00Z",
    ...overrides,
  };
}

/** The runs list is whatever the test supplies; so is the protocols list, empty by default. */
function serve(runs: RunResponse[], protocols: unknown[] = []) {
  vi.mocked(customInstance).mockImplementation(async ({ url, params }) => {
    if (url.endsWith("/engines")) return [{ id: "xgb", name: "XGBoost" }];
    if (url.endsWith("/runs")) {
      const statuses = params?.status as string[] | undefined;
      const items = statuses ? runs.filter((run) => statuses.includes(run.status)) : runs;
      return { items, next_cursor: null };
    }
    if (url.endsWith("/protocols")) return { items: protocols, next_cursor: null };
    return { items: [], next_cursor: null };
  });
}

describe("the In training section", () => {
  beforeEach(() => {
    vi.mocked(customInstance).mockReset();
    nav.replace.mockReset();
    nav.query = "";
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

  it("leaves out finished runs and lists stopped ones to resume or delete", async () => {
    serve([
      trainingRun({ id: "run-live", name: "still going" }),
      trainingRun({ id: "run-done", status: "ready", name: "finished model" }),
      trainingRun({ id: "run-bad", status: "failed", name: "failed model" }),
      trainingRun({ id: "run-off", status: "cancelled", name: "canceled model" }),
    ]);
    render(<ProtocolList />, { wrapper: Wrapper });

    expect(await screen.findByText("still going")).toBeInTheDocument();
    expect(screen.queryByText("finished model")).not.toBeInTheDocument();
    const failed = await screen.findByRole("link", { name: /failed model/ });
    expect(failed).toHaveAttribute("href", "/runs/run-bad");
    expect(failed).toHaveTextContent("Failed · open to resume or delete");
    expect(screen.getByRole("link", { name: /canceled model/ })).toHaveTextContent("Canceled");
  });

  it("renders no section when nothing is live", async () => {
    serve([trainingRun({ status: "ready" })]);
    render(<ProtocolList />, { wrapper: Wrapper });

    expect(await screen.findByText("No protocols yet")).toBeInTheDocument();
    expect(screen.queryByText("In training")).not.toBeInTheDocument();
  });
});

describe("the Mine filter and creators", () => {
  beforeEach(() => {
    vi.mocked(customInstance).mockReset();
    nav.replace.mockReset();
    nav.query = "";
  });

  it("asks for mine=true and marks the toggle pressed when the URL says so", async () => {
    nav.query = "mine=1";
    serve([]);
    render(<ProtocolList />, { wrapper: Wrapper });

    expect(screen.getByRole("button", { name: "Created by me" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(
      within(screen.getByRole("group", { name: "Protocol owner" })).getByRole("button", {
        name: "All",
      }),
    ).toHaveAttribute("aria-pressed", "false");
    await waitFor(() =>
      expect(customInstance).toHaveBeenCalledWith(
        expect.objectContaining({
          url: "/api/v1/protocols",
          params: expect.objectContaining({ mine: true }),
        }),
      ),
    );
    expect(await screen.findByText("You have not trained a protocol yet.")).toBeInTheDocument();
  });

  it("writes mine=1 to the URL when Mine is clicked", async () => {
    serve([]);
    render(<ProtocolList />, { wrapper: Wrapper });
    fireEvent.click(screen.getByRole("button", { name: "Created by me" }));
    expect(nav.replace).toHaveBeenCalledWith("/protocols?mine=1");
  });

  it("shows the creator, engine and score on a card", async () => {
    serve(
      [
        trainingRun({
          id: "run-p1",
          status: "ready",
          protocol_id: "p1",
          metrics: {
            targets: [
              { column: "pIC50", primary_metric: "rmse", value: 0.52, baseline_value: 0.61 },
            ],
          },
        }),
      ],
      [
        {
          id: "p1",
          name: "hERG",
          status: "draft",
          engine_id: "xgb",
          readouts: [{ name: "pIC50" }],
          protocol_version: 1,
          created_at: "2026-10-03T12:00:00Z",
          created_by: "user-1",
          folder_id: null,
        },
      ],
    );
    render(<ProtocolList />, { wrapper: Wrapper });
    expect(await screen.findByRole("img", { name: "Ada" })).toBeInTheDocument();
    expect(await screen.findByText("XGBoost")).toBeInTheDocument();
    expect(await screen.findByText("0.520")).toBeInTheDocument();
    expect(screen.getByText("0.090 better")).toBeInTheDocument();
    expect(screen.getByText("Draft")).toBeInTheDocument();
    expect(customInstance).toHaveBeenCalledWith(
      expect.objectContaining({
        url: "/api/v1/runs",
        params: { kind: "training", limit: 200 },
      }),
    );
  });
});
