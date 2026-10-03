import { describe, expect, it } from "vitest";
import { pollInterval } from "./query-defaults";

describe("pollInterval", () => {
  const q = (status: string, dataStatus?: string) =>
    ({ state: { status, data: dataStatus ? { status: dataStatus } : undefined } }) as never;

  it("keeps polling a live run", () => expect(pollInterval(q("success", "running"))).toBe(2000));
  it("stops on a terminal status", () => expect(pollInterval(q("success", "failed"))).toBe(false));
  it("stops when the request itself fails", () => expect(pollInterval(q("error"))).toBe(false));
});
