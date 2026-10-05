import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useMemberName } from "./use-workspace-members";

const listMembers = vi.fn();
vi.mock("./config", () => ({ getDuarClient: () => ({ listMembers }) }));

const client = new QueryClient();
function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("useMemberName", () => {
  beforeEach(() => {
    listMembers.mockReset();
    client.clear();
  });

  it("resolves a user id to a name, and an unknown one to undefined", async () => {
    listMembers.mockResolvedValue([{ user_id: "u1", name: "Siddhant Rath" }]);
    const { result } = renderHook(() => useMemberName(), { wrapper });
    await waitFor(() => expect(result.current("u1")).toBe("Siddhant Rath"));
    expect(result.current("u2")).toBeUndefined();
  });

  it("is undefined, and does not throw, when Duar cannot be reached", async () => {
    listMembers.mockImplementation(() => Promise.reject(new Error("down")));
    const { result } = renderHook(() => useMemberName(), { wrapper });
    await waitFor(() => expect(listMembers).toHaveBeenCalled());
    // Let the rejection settle into the query's error state.
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(result.current("u1")).toBeUndefined();
  });
});
