import { customInstance } from "@/shared/lib/api/custom-instance";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RunList } from "./run-list";

vi.mock("@/shared/lib/api/custom-instance", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/shared/lib/api/custom-instance")>()),
  customInstance: vi.fn(),
}));

const nav = vi.hoisted(() => ({ replace: vi.fn(), query: "" }));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: nav.replace }),
  usePathname: () => "/runs",
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

function run(id: string, createdAt: string, name: string | null) {
  return {
    id,
    kind: "prediction",
    status: "ready",
    protocol_id: "p1",
    name,
    requested_by: "user-1",
    metrics: { scored_rows: 12 },
    source: null,
    created_at: createdAt,
  };
}

function serve(runs: unknown[]) {
  vi.mocked(customInstance).mockImplementation(async ({ url }) => {
    if (url.endsWith("/runs")) return { items: runs, next_cursor: null };
    if (url.endsWith("/folders")) return { items: [], can_edit: true };
    return {
      items: [{ id: "p1", name: "hERG", status: "published" }],
      next_cursor: null,
    };
  });
}

function runsCall() {
  return vi.mocked(customInstance).mock.calls.find(([arg]) => arg.url.endsWith("/runs"))?.[0];
}

describe("RunList", () => {
  beforeEach(() => {
    vi.mocked(customInstance).mockReset();
    nav.replace.mockReset();
    nav.query = "";
  });

  it("groups runs under day headers and shows name, protocol and member", async () => {
    serve([
      run("r1", "2026-10-05T18:00:00Z", "batch-7"),
      run("r2", "2026-10-05T17:00:00Z", null),
      run("r3", "2026-10-03T17:00:00Z", "older"),
    ]);
    render(<RunList />, { wrapper: Wrapper });

    expect(await screen.findByRole("heading", { name: /Oct 5, 2026/ })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Oct 3, 2026/ })).toBeInTheDocument();
    const row = screen.getByRole("link", { name: /batch-7/ });
    expect(row).toHaveAttribute("href", "/runs/r1");
    expect(row).toHaveTextContent("hERG");
    expect(row).toHaveTextContent("Ada");
    // A run without a name falls back to its protocol's name.
    expect(document.querySelector('a[href="/runs/r2"]')).toHaveTextContent("hERG · 12 compounds");
  });

  it("has Mine on by default and requests mine=true", async () => {
    serve([]);
    render(<RunList />, { wrapper: Wrapper });
    const owner = screen.getByRole("group", { name: "Run owner" });
    expect(within(owner).getByRole("button", { name: "Mine" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    await waitFor(() =>
      expect(runsCall()?.params).toEqual(expect.objectContaining({ mine: true })),
    );
  });

  it("sends the status filter from the URL as a list", async () => {
    nav.query = "status=pending,running";
    serve([]);
    render(<RunList />, { wrapper: Wrapper });
    await waitFor(() =>
      expect(runsCall()?.params).toEqual(
        expect.objectContaining({ status: ["pending", "running"] }),
      ),
    );
  });

  it("writes the status choice to the URL", async () => {
    serve([]);
    render(<RunList />, { wrapper: Wrapper });
    const trigger = screen.getByRole("combobox", { name: "Status" });
    fireEvent.keyDown(trigger, { key: "Enter" });
    fireEvent.click(await screen.findByRole("option", { name: "Failed" }));
    expect(nav.replace).toHaveBeenCalledWith("/runs?status=failed");
  });

  it("offers to clear filters when none match", async () => {
    nav.query = "mine=0&q=zzz";
    serve([]);
    render(<RunList />, { wrapper: Wrapper });
    expect(await screen.findByText("No runs match these filters.")).toBeInTheDocument();
    const clear = screen.getAllByRole("button", { name: "Clear filters" });
    fireEvent.click(clear[clear.length - 1]);
    expect(nav.replace).toHaveBeenCalledWith("/runs");
  });
});
