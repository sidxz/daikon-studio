import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { isComputing, profileRefetchInterval } from "../hooks/use-datasets";
import { ProfileComputing, elapsedLabel } from "./profile-computing";

describe("elapsedLabel", () => {
  const start = Date.parse("2026-10-03T20:00:00Z");
  it("counts seconds, then minutes and seconds, then hours and minutes", () => {
    expect(elapsedLabel(start, start + 45_000)).toBe("45 s");
    expect(elapsedLabel(start, start + 192_000)).toBe("3 min 12 s");
    expect(elapsedLabel(start, start + 3_720_000)).toBe("1 h 2 min");
  });
  it("never shows a negative time when the clocks disagree", () => {
    expect(elapsedLabel(start, start - 5_000)).toBe("0 s");
  });
});

describe("profile polling", () => {
  const computing = { status: "computing", started_at: "2026-10-03T20:00:00Z", compounds: 412_345 };
  it("recognizes the computing answer", () => {
    expect(isComputing(computing as never)).toBe(true);
    expect(isComputing({ scaffolds: {} } as never)).toBe(false);
    expect(isComputing(undefined)).toBe(false);
  });
  it("polls only while computing", () => {
    expect(profileRefetchInterval(computing as never)).toBe(3000);
    expect(profileRefetchInterval({ scaffolds: {} } as never)).toBe(false);
  });
});

describe("ProfileComputing", () => {
  it("says what is being computed, for how many compounds, and that reloading is safe", () => {
    render(
      <ProfileComputing
        startedAt={new Date(Date.now() - 65_000).toISOString()}
        compounds={412_345}
      />,
    );
    expect(screen.getByText("Profiling 412,345 compounds")).toBeInTheDocument();
    expect(screen.getByText(/Running for 1 min 5 s/)).toBeInTheDocument();
    expect(screen.getByText(/does not restart it/)).toBeInTheDocument();
  });
});
