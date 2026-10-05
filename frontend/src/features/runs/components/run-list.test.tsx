import { customInstance } from "@/shared/lib/api/custom-instance";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { dayLabel } from "../lib/group-runs";
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

function run(id: string, createdAt: string, name: string | null, status = "ready") {
  return {
    id,
    kind: "prediction",
    status,
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

  it("groups runs under day headers and shows name, protocol, compounds and member", async () => {
    const today = new Date();
    const daysAgo = (days: number) =>
      new Date(today.getFullYear(), today.getMonth(), today.getDate() - days, 12).toISOString();
    const older = daysAgo(10);
    serve([
      run("r1", today.toISOString(), "batch-7"),
      run("r2", today.toISOString(), null),
      run("r3", daysAgo(1), "last night"),
      run("r4", older, "older"),
    ]);
    render(<RunList />, { wrapper: Wrapper });

    expect(await screen.findByRole("heading", { name: "Today" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Yesterday" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: dayLabel(older, new Date()) })).toBeInTheDocument();
    const row = screen.getByRole("link", { name: /batch-7/ });
    expect(row).toHaveAttribute("href", "/runs/r1");
    expect(row).toHaveTextContent("hERG");
    expect(row).toHaveTextContent("12 compounds");
    expect(within(row).getByRole("img", { name: "Ada" })).toBeInTheDocument();
    // A run without a name falls back to its protocol's name, and is not repeated beside it.
    const unnamed = document.querySelector('a[href="/runs/r2"]') as HTMLElement;
    expect(unnamed.textContent?.match(/hERG/g)).toHaveLength(1);
    expect(unnamed.textContent).not.toContain("·");
  });

  it("shows each status as a dot and a word, with no filled Ready pill", async () => {
    const at = new Date().toISOString();
    serve([
      run("r1", at, "a", "ready"),
      run("r2", at, "b", "running"),
      run("r3", at, "c", "pending"),
      run("r4", at, "d", "failed"),
      run("r5", at, "e", "cancelled"),
    ]);
    render(<RunList />, { wrapper: Wrapper });

    expect(await screen.findByText("Ready")).toBeInTheDocument();
    expect(screen.getByText("Running")).toBeInTheDocument();
    expect(screen.getByText("Queued")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toHaveClass("text-destructive");
    expect(screen.getByText("Canceled")).toBeInTheDocument();
    expect(screen.getByText("Ready").closest('[data-slot="badge"]')).toBeNull();
    expect(document.querySelectorAll(".motion-safe\\:animate-status-breathe")).toHaveLength(2);
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
    expect(screen.getByRole("button", { name: "Clear" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(nav.replace).toHaveBeenCalledWith("/runs");
  });
});
