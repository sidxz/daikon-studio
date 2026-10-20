import { customInstance } from "@/shared/lib/api/custom-instance";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Runner } from "../types";
import { RunnerList } from "./runner-list";

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

function runner(overrides: Partial<Runner> = {}): Runner {
  return {
    id: "r1",
    name: "gpu-01",
    lanes: ["gpu"],
    online: true,
    last_seen_at: "2026-10-20T09:00:00Z",
    revoked: false,
    current_run_id: null,
    created_at: "2026-10-01T00:00:00Z",
    build_version: null,
    build_git_sha: null,
    build_date: null,
    ...overrides,
  };
}

function serve(runners: Runner[]) {
  vi.mocked(customInstance).mockImplementation(async ({ url, method }) => {
    if (url.endsWith("/runners") && method === "GET") return runners;
    return undefined;
  });
}

function row(name: string) {
  return screen.getByText(name).closest("tr") as HTMLElement;
}

describe("RunnerList", () => {
  beforeEach(() => {
    vi.mocked(customInstance).mockReset();
  });

  it("shows the version an agent reported, with the commit beside it", async () => {
    serve([
      runner({
        build_version: "0.8.0",
        build_git_sha: "0123456789abcdef0123456789abcdef01234567",
        build_date: "2026-10-20T09:00:00Z",
      }),
    ]);
    render(<RunnerList />, { wrapper: Wrapper });

    await waitFor(() => expect(within(row("gpu-01")).getByText("0.8.0")).toBeTruthy());
    expect(within(row("gpu-01")).getByText("0123456")).toBeTruthy();
    expect(
      within(row("gpu-01")).getByTitle(
        "Commit 0123456789abcdef0123456789abcdef01234567, built 2026-10-20T09:00:00Z",
      ),
    ).toBeTruthy();
  });

  it("says unknown for a runner that has reported no build", async () => {
    serve([runner()]);
    render(<RunnerList />, { wrapper: Wrapper });

    await waitFor(() => expect(within(row("gpu-01")).getByText("unknown")).toBeTruthy());
    expect(
      within(row("gpu-01")).getByTitle(
        "This runner has not reported a build. It will the next time its agent starts.",
      ),
    ).toBeTruthy();
  });
});
