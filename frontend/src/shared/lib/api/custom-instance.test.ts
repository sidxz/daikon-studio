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

describe("error messages", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("shows a domain error's own message instead of a bare status", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({ error: "AuthorizationError", message: "Requires admin role or higher" }),
            { status: 403 },
          ),
      ),
    );
    await expect(customInstance({ url: "/api/v1/runners", method: "POST" })).rejects.toMatchObject({
      status: 403,
      message: "Requires admin role or higher",
    });
  });

  it("still flattens FastAPI's validation shape", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(JSON.stringify({ detail: [{ loc: ["body", "name"], msg: "too long" }] }), {
            status: 422,
          }),
      ),
    );
    await expect(customInstance({ url: "/api/v1/runners", method: "POST" })).rejects.toMatchObject({
      message: "API error: 422 — body.name: too long",
    });
  });
});
