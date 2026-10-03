import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { shouldRenew, useSessionRenewal } from "./session-renewal";

const silentLogin = vi.fn(() => true);
let onUnauthorized: (() => void) | null = null;

vi.mock("@/shared/lib/auth/config", () => ({
  getDuarClient: () => ({ silentLogin }),
  idpTokenExpiresAt: () => null,
}));
vi.mock("@/shared/lib/api/custom-instance", () => ({
  setUnauthorizedHandler: (handler: (() => void) | null) => {
    onUnauthorized = handler;
  },
}));

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
    for (let load = 0; load < 3; load++) {
      const { unmount } = renderHook(() => useSessionRenewal());
      onUnauthorized?.();
      unmount();
    }
    expect(silentLogin).toHaveBeenCalledTimes(1);
  });
});
