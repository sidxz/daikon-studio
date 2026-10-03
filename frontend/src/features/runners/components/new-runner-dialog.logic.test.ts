import { setApiBaseUrl } from "@/shared/lib/api/custom-instance";
import { describe, expect, it } from "vitest";
import { apiOrigin, runCommand } from "./new-runner-dialog";

const runner = {
  id: "r1",
  name: "lab-workstation-1",
  lanes: ["default"],
  token: "drt_secret",
  created_at: "2026-08-04T00:00:00Z",
};

describe("apiOrigin", () => {
  it("returns the origin of an absolute base URL", () => {
    setApiBaseUrl("http://localhost:8002");
    expect(apiOrigin()).toBe("http://localhost:8002");
  });

  it("falls back to the page origin for a relative base instead of throwing", () => {
    // APP_API_BASE_URL=/api -- `new URL("/api")` throws with no base argument.
    // Important 5, final review: must not throw during the token-reveal render.
    setApiBaseUrl("/api");
    expect(() => apiOrigin()).not.toThrow();
    expect(apiOrigin()).toBe(window.location.origin);
    setApiBaseUrl("http://localhost:8002"); // restore, other tests may share this module
  });
});

describe("runCommand", () => {
  const images = { cpu: "ghcr.io/lab/daikon-studio/api", gpu: "registry.lab/daikon-runner:gpu" };

  it("overrides CMD for the cpu image, whose default CMD serves the API", () => {
    setApiBaseUrl("http://localhost:8002");
    const command = runCommand({ ...runner, lanes: ["default"] }, images);
    expect(command).toContain("ghcr.io/lab/daikon-studio/api");
    expect(command).toContain("python -m daikonstudio.infrastructure.runner");
    expect(command).not.toContain("--gpus all");
  });

  it("adds --gpus all for the gpu image and does not override its CMD", () => {
    setApiBaseUrl("http://localhost:8002");
    const command = runCommand({ ...runner, lanes: ["gpu"] }, images);
    expect(command).toContain("registry.lab/daikon-runner:gpu");
    expect(command).toContain("--gpus all");
    expect(command).not.toContain("python -m daikonstudio.infrastructure.runner");
  });
});
