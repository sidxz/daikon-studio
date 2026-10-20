import { ApiError, customInstance } from "@/shared/lib/api/custom-instance";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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

  it("deletes a runner after the confirmation, and says how delete differs from revoke", async () => {
    serve([runner()]);
    render(<RunnerList />, { wrapper: Wrapper });

    fireEvent.click(
      await waitFor(() => within(row("gpu-01")).getByRole("button", { name: "Delete" })),
    );

    expect(screen.getByText("Delete gpu-01?")).toBeTruthy();
    const dialog = screen.getByRole("alertdialog");
    expect(dialog.textContent).toContain("cannot be undone");
    expect(dialog.textContent).toContain("revoke that one instead");

    fireEvent.click(screen.getByRole("button", { name: "Delete runner" }));

    await waitFor(() =>
      expect(vi.mocked(customInstance)).toHaveBeenCalledWith(
        expect.objectContaining({ url: "/api/v1/runners/r1", method: "DELETE" }),
      ),
    );
  });

  it("does not call the API until the dialog is confirmed", async () => {
    serve([runner()]);
    render(<RunnerList />, { wrapper: Wrapper });

    fireEvent.click(
      await waitFor(() => within(row("gpu-01")).getByRole("button", { name: "Delete" })),
    );
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    expect(vi.mocked(customInstance).mock.calls.some(([arg]) => arg.method === "DELETE")).toBe(
      false,
    );
  });

  it("keeps the dialog open when the server refuses the delete", async () => {
    /** The 409 for a runner whose runs name it: the shared mutation toast
     *  carries the reason, so the dialog must not close as if it had worked. */
    vi.mocked(customInstance).mockImplementation(async ({ url, method }) => {
      if (url.endsWith("/runners") && method === "GET") return [runner()];
      throw new ApiError("'gpu-01' has claimed 3 runs, and that history names it.", 409, {});
    });
    render(<RunnerList />, { wrapper: Wrapper });

    fireEvent.click(
      await waitFor(() => within(row("gpu-01")).getByRole("button", { name: "Delete" })),
    );
    fireEvent.click(screen.getByRole("button", { name: "Delete runner" }));

    await waitFor(() =>
      expect(vi.mocked(customInstance).mock.calls.some(([arg]) => arg.method === "DELETE")).toBe(
        true,
      ),
    );
    expect(screen.getByText("Delete gpu-01?")).toBeTruthy();
  });

  it("offers delete but not revoke for an already revoked runner", async () => {
    serve([runner({ revoked: true, online: false })]);
    render(<RunnerList />, { wrapper: Wrapper });

    const actions = await waitFor(() => row("gpu-01"));
    expect(within(actions).getByRole("button", { name: "Delete" })).toBeTruthy();
    expect(within(actions).queryByRole("button", { name: "Revoke" })).toBeNull();
  });
});
