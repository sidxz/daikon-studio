import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, customInstance, setUnauthorizedHandler } from "./custom-instance";

vi.mock("@/shared/lib/auth/config", () => ({
  getDuarClient: () => ({ isAuthenticated: false, getHeaders: () => ({}) }),
}));

const expired = () =>
  new Response(JSON.stringify({ detail: "IdP token expired" }), { status: 401 });

describe("401 handling", () => {
  afterEach(() => {
    setUnauthorizedHandler(null);
    vi.unstubAllGlobals();
  });

  it("notifies the handler once and throws a silent ApiError", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => expired()));
    const handler = vi.fn();
    setUnauthorizedHandler(handler);

    await expect(customInstance({ url: "/api/v1/runs", method: "GET" })).rejects.toMatchObject({
      status: 401,
      silent: true,
    });
    await expect(customInstance({ url: "/api/v1/runs", method: "GET" })).rejects.toBeInstanceOf(
      ApiError,
    );
    expect(handler).toHaveBeenCalledTimes(1);
  });
});
