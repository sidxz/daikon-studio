import type { ConditionResponse, EngineManifestResponse } from "@/shared/lib/api/model";

// Generated DTOs, re-exported under domain names. Never redeclare their shape --
// `pnpm generate:api` is what keeps them true.
export type Engine = EngineManifestResponse;
export type Condition = ConditionResponse;

/**
 * `ConditionResponse.type` is a bare `string` in the contract, not a closed
 * enum, so codegen cannot give us a union. These are the five values the
 * backend's `ConditionType` can emit.
 */
export type ConditionType = "string" | "integer" | "number" | "enum" | "bool";

/**
 * Engines advertise task types; a Dataset's target declares a kind. The two
 * vocabularies differ and the mapping lives only server-side -- this is the one
 * piece of hardcoded knowledge the self-describing catalogue failed to
 * eliminate, so it lives in exactly one place.
 */
export const TASK_FOR_TARGET_KIND = {
  numeric: "regression",
  binary: "binary_classification",
} as const;

export const TASK_LABELS: Record<string, string> = {
  regression: "Regression",
  binary_classification: "Binary classification",
};

/** Engines that can be trained on a dataset with this target kind. */
export function enginesForTargetKind(
  engines: Engine[],
  kind: keyof typeof TASK_FOR_TARGET_KIND,
): Engine[] {
  const task = TASK_FOR_TARGET_KIND[kind];
  return engines.filter((engine) => engine.tasks.includes(task));
}
