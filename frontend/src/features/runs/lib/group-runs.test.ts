import { describe, expect, it } from "vitest";
import { groupRunsByDay } from "./group-runs";

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
