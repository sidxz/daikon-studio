import { describe, expect, it } from "vitest";
import { groupRuns } from "./group-runs";

describe("groupRuns", () => {
  it("groups by protocol, then day, keeping newest-first order", () => {
    const day = (iso: string) => iso.slice(0, 10);
    const runs = [
      { id: "1", protocol_id: "herg", created_at: "2026-10-05T10:46:00Z" },
      { id: "2", protocol_id: "herg", created_at: "2026-10-05T10:37:00Z" },
      { id: "3", protocol_id: "esol", created_at: "2026-10-04T09:37:00Z" },
      { id: "4", protocol_id: "herg", created_at: "2026-10-03T08:00:00Z" },
    ];
    expect(
      groupRuns(runs, day).map((group) => ({
        protocolId: group.protocolId,
        days: group.days.map((d) => [d.day, d.runs.map((run) => run.id)]),
      })),
    ).toEqual([
      {
        protocolId: "herg",
        days: [
          ["2026-10-05", ["1", "2"]],
          ["2026-10-03", ["4"]],
        ],
      },
      { protocolId: "esol", days: [["2026-10-04", ["3"]]] },
    ]);
  });
});
