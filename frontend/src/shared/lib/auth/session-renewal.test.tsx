import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { shouldRenew, useSessionRenewal } from "./session-renewal";

const silentLogin = vi.fn(() => true);
let onUnauthorized: (() => boolean) | null = null;

vi.mock("@/shared/lib/auth/config", () => ({
  getDuarClient: () => ({ silentLogin }),
  idpTokenExpiresAt: () => null,
}));
vi.mock("@/shared/lib/api/custom-instance", () => ({
  setUnauthorizedHandler: (handler: (() => boolean) | null) => {
    onUnauthorized = handler;
  },
}));

// The hook reads the QueryClient (to hold off a proactive renewal while a
// mutation is in flight), so it renders inside a provider, as in the app.
function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>;
}

describe("shouldRenew", () => {
  it("is false with no token or a token far from expiry", () => {
    expect(shouldRenew(null, 1_000_000)).toBe(false);
    expect(shouldRenew(1_000_000 + 3_600_000, 1_000_000)).toBe(false);
  });

  it("is true inside the 90 s window and after expiry", () => {
    expect(shouldRenew(1_000_000 + 60_000, 1_000_000)).toBe(true);
    expect(shouldRenew(1_000_000 - 1, 1_000_000)).toBe(true);
  });
});

describe("a 401 on every page load", () => {
  afterEach(() => sessionStorage.clear());

  it("starts one renewal, not a redirect loop through the IdP", () => {
    // The renewal is a full-page round trip; each load gets the 401 again.
    const started: boolean[] = [];
    for (let load = 0; load < 3; load++) {
      const { unmount } = renderHook(() => useSessionRenewal(), { wrapper });
      started.push(onUnauthorized?.() ?? false);
      unmount();
    }
    expect(silentLogin).toHaveBeenCalledTimes(1);
    // The handler reports what it did, so the 401 latch knows whether to re-arm.
    expect(started).toEqual([true, false, false]);
  });
});
