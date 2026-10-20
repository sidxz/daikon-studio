import { describe, expect, it } from "vitest";
import { buildDetail, shortSha } from "./format-build";

describe("buildDetail", () => {
  it("names the commit and the build date", () => {
    expect(buildDetail({ build_git_sha: "abc1234", build_date: "2026-10-20T09:00:00Z" })).toBe(
      "Commit abc1234, built 2026-10-20T09:00:00Z",
    );
  });

  it("says unknown for a field the runner did not report", () => {
    expect(buildDetail({ build_git_sha: null, build_date: null })).toBe(
      "Commit unknown, built unknown",
    );
  });
});

describe("shortSha", () => {
  it("abbreviates a full sha to seven characters", () => {
    expect(shortSha("0123456789abcdef0123456789abcdef01234567")).toBe("0123456");
  });

  it("leaves anything that is not a full sha alone", () => {
    expect(shortSha("abc1234")).toBe("abc1234");
    expect(shortSha("unknown")).toBe("unknown");
  });
});
