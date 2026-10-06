import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { parseLocalDay, runDateBounds } from "./run-date-range";

describe("run date range", () => {
  beforeEach(() => vi.stubEnv("TZ", "America/Chicago"));
  afterEach(() => vi.unstubAllEnvs());

  it("includes the entire selected end day in local time", () => {
    expect(runDateBounds("2026-10-05", "2026-10-05")).toEqual({
      createdFrom: "2026-10-05T05:00:00.000Z",
      createdBefore: "2026-10-06T05:00:00.000Z",
    });
  });
  it("uses calendar days across both daylight saving transitions", () => {
    expect(runDateBounds("2026-03-08", "2026-03-08")).toEqual({
      createdFrom: "2026-03-08T06:00:00.000Z",
      createdBefore: "2026-03-09T05:00:00.000Z",
    });
    expect(runDateBounds("2026-11-01", "2026-11-01")).toEqual({
      createdFrom: "2026-11-01T05:00:00.000Z",
      createdBefore: "2026-11-02T06:00:00.000Z",
    });
  });
  it("allows open-ended ranges and rejects invalid calendar days", () => {
    expect(runDateBounds("", "2026-10-05").createdFrom).toBeUndefined();
    expect(runDateBounds("2026-10-05", "").createdBefore).toBeUndefined();
    expect(parseLocalDay("2026-02-30")).toBeNull();
    expect(parseLocalDay("nonsense")).toBeNull();
  });
});
