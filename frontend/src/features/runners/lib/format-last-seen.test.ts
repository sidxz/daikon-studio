import { describe, expect, it } from "vitest";
import { formatLastSeen } from "./format-last-seen";

const now = Date.parse("2026-08-04T12:00:00Z");

describe("formatLastSeen", () => {
  it("reports a runner that has never checked in", () => {
    expect(formatLastSeen(null, now)).toBe("Never");
  });

  it("rounds a fresh heartbeat down to just now", () => {
    expect(formatLastSeen(new Date(now - 2_000).toISOString(), now)).toBe("Just now");
  });

  it("counts seconds within the last minute", () => {
    expect(formatLastSeen(new Date(now - 30_000).toISOString(), now)).toBe("30s ago");
  });

  it("counts minutes within the last hour", () => {
    expect(formatLastSeen(new Date(now - 5 * 60_000).toISOString(), now)).toBe("5m ago");
  });

  it("counts hours within the last day", () => {
    expect(formatLastSeen(new Date(now - 3 * 3_600_000).toISOString(), now)).toBe("3h ago");
  });

  it("falls back to days for anything older", () => {
    expect(formatLastSeen(new Date(now - 2 * 86_400_000).toISOString(), now)).toBe("2d ago");
  });
});
