import { ApiError, customInstance } from "@/shared/lib/api/custom-instance";
import { showError } from "@/shared/lib/toast";
import { QueryProvider } from "@/shared/providers/query-provider";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useActiveTrainingRuns, usePublishProtocol } from "./use-protocols";

vi.mock("@/shared/lib/toast", () => ({ showError: vi.fn(), showSuccess: vi.fn() }));
vi.mock("@/shared/lib/api/custom-instance", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/shared/lib/api/custom-instance")>()),
  customInstance: vi.fn(),
}));

const invalidate = vi.spyOn(QueryClient.prototype, "invalidateQueries");

async function publishFailingWith(error: ApiError) {
  vi.mocked(customInstance).mockRejectedValueOnce(error);
  const { result } = renderHook(() => usePublishProtocol(), { wrapper: QueryProvider });
  act(() => result.current.mutate("protocol-1"));
  await waitFor(() => expect(result.current.isError).toBe(true));
}

describe("a failed publish", () => {
  afterEach(() => {
    vi.mocked(showError).mockClear();
    invalidate.mockClear();
  });

  it("is silent when someone published it first, and refreshes the stale view", async () => {
    await publishFailingWith(new ApiError("Request failed (423)", 423, undefined));
    expect(showError).not.toHaveBeenCalled();
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["protocol", "protocol-1"] });
  });

  it("toasts any other failure exactly once", async () => {
    await publishFailingWith(new ApiError("Request failed (500)", 500, undefined));
    expect(showError).toHaveBeenCalledExactlyOnceWith("Request failed (500)");
  });
});

describe("useActiveTrainingRuns", () => {
  afterEach(() => invalidate.mockClear());

  it("refreshes the Protocols list once a live run finishes", async () => {
    const run = (status: string) => ({ items: [{ id: "run-1", status }], next_cursor: null });
    vi.mocked(customInstance).mockResolvedValueOnce(run("running")).mockResolvedValue(run("ready"));
    const queryClient = new QueryClient();
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
    const { result } = renderHook(() => useActiveTrainingRuns(), { wrapper });

    await waitFor(() => expect(result.current).toHaveLength(1));
    expect(invalidate).not.toHaveBeenCalledWith({ queryKey: ["protocols"] });

    await act(() => queryClient.refetchQueries({ queryKey: ["protocol-runs"] }));
    await waitFor(() => expect(result.current).toHaveLength(0));
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["protocols"] });
  });
});
