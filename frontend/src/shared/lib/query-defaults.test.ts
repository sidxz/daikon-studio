import { describe, expect, it } from "vitest";
import { RUN_RETRY_POLL_MS, STALE_TIME, mapStaleTime, pollInterval } from "./query-defaults";

describe("pollInterval", () => {
  const q = (status: string, dataStatus?: string) =>
    ({ state: { status, data: dataStatus ? { status: dataStatus } : undefined } }) as never;

  it("keeps polling a live run", () => expect(pollInterval(q("success", "running"))).toBe(2000));
  it("stops on a terminal status", () => expect(pollInterval(q("success", "failed"))).toBe(false));
  it("stops when the request fails with nothing loaded", () =>
    expect(pollInterval(q("error"))).toBe(false));
  it("backs off, rather than stopping, when a refetch fails with data on screen", () =>
    expect(pollInterval(q("error", "running"))).toBe(RUN_RETRY_POLL_MS));
});

describe("mapStaleTime", () => {
  const q = (status?: string) => ({ state: { data: status ? { status } : undefined } }) as never;

  it("never trusts a 'missing' map: training draws it last, so it can appear a minute later", () =>
    expect(mapStaleTime(q("missing"))).toBe(0));
  it("keeps a ready map for long; it never changes", () =>
    expect(mapStaleTime(q("ready"))).toBe(STALE_TIME.LONG));
  it("treats a not-yet-loaded map like a ready one", () =>
    expect(mapStaleTime(q())).toBe(STALE_TIME.LONG));
});
