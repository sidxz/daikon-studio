import type { CreatedRunnerResponse, RunnerResponse } from "@/shared/lib/api/model";

// Generated DTOs, re-exported under domain names -- never redeclare their shape.
export type Runner = RunnerResponse;
/** Only the create response carries `token`, and only once. */
export type CreatedRunner = CreatedRunnerResponse;

/**
 * Lanes a runner can serve. `EngineManifestResponse.lane` names the lane each
 * engine needs, but the New-runner dialog has to offer a lane before any engine
 * is involved, so this mirrors the two values that field takes today
 * (`backend/src/daikonstudio/application/engines/manifest.py`). Add a lane here
 * the day a third one ships.
 */
export const KNOWN_LANES = ["default", "gpu"] as const;
export type Lane = (typeof KNOWN_LANES)[number];

export const LANE_LABELS: Record<string, string> = {
  default: "Default",
  gpu: "GPU",
};
