import { describe, expect, it } from "vitest";
import { dayLabel, groupRunsByDay } from "./group-runs";

describe("groupRunsByDay", () => {
  it("groups by calendar day, keeping the incoming newest-first order", () => {
    const day = (iso: string) => iso.slice(0, 10);
    const runs = [
      { id: "1", created_at: "2026-10-05T10:46:00Z" },
      { id: "2", created_at: "2026-10-05T10:37:00Z" },
      { id: "3", created_at: "2026-10-04T09:37:00Z" },
    ];
    expect(groupRunsByDay(runs, day).map((g) => [g.day, g.runs.map((run) => run.id)])).toEqual([
      ["2026-10-05", ["1", "2"]],
      ["2026-10-04", ["3"]],
    ]);
  });
});

describe("dayLabel", () => {
  // Built from local dates, so the test holds in any time zone.
  const now = new Date(2026, 9, 5, 15, 0);
  const at = (year: number, month: number, day: number, hour = 9) =>
    new Date(year, month, day, hour).toISOString();

  it("says Today and Yesterday by calendar day, not by 24 hours", () => {
    expect(dayLabel(at(2026, 9, 5, 0), now)).toBe("Today");
    expect(dayLabel(at(2026, 9, 4, 23), now)).toBe("Yesterday");
    expect(dayLabel(at(2026, 9, 4, 0), now)).toBe("Yesterday");
  });

  it("names an older day this year without the year", () => {
    expect(dayLabel(at(2026, 9, 3), now)).toBe("Sat, Oct 3");
  });

  it("adds the year for another year", () => {
    expect(dayLabel(at(2025, 9, 3), now)).toBe("Fri, Oct 3, 2025");
  });
});
