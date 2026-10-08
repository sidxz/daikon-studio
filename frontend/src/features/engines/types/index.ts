import type { ConditionResponse, EngineManifestResponse, TargetBody } from "@/shared/lib/api/model";

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

/**
 * What each pretrained weight set fixes, mirroring its checkpoint's own saved
 * `hyper_parameters`.
 *
 * The second piece of hardcoded engine knowledge in this app, and it is here
 * for the same reason as TASK_FOR_TARGET_KIND: the manifest has no way to say
 * "this condition makes those two inert". Without it the form would accept a
 * `depth` the fit silently ignores, and the Scorecard would then report a
 * setting the model never used, the exact dishonesty the Scorecard exists to
 * prevent. The form submits these values, so the record stays true.
 *
 * ponytail: two constants for one weight set. If a second one lands, move this
 * onto ConditionSpec as a `pinned_by` field so the catalogue stays
 * self-describing.
 */
export const PINNED_BY_PRETRAINED: Record<string, Record<string, number>> = {
  CheMeleon: { message_hidden_dim: 2048, depth: 6 },
};

type HasKind = Pick<TargetBody, "kind">;

/** The distinct tasks these targets ask of an engine. */
export function tasksForTargets(targets: HasKind[]): string[] {
  return [...new Set(targets.map((target) => TASK_FOR_TARGET_KIND[target.kind]))];
}

/**
 * Engines that can train on every one of these targets. An engine must support
 * each target's task; a joint engine (`supports_multitask`) learns them all in one
 * model, so the server refuses it a dataset that mixes measured values and
 * active/inactive labels, and so does this list.
 */
export function enginesForTargets(
  engines: Engine[],
  targets: HasKind[],
  structureKind?: string,
): Engine[] {
  const kinds = new Set(targets.map((target) => target.kind));
  const tasks = tasksForTargets(targets);
  return engines.filter(
    (engine) =>
      tasks.every((task) => engine.tasks.includes(task)) &&
      (!engine.supports_multitask || kinds.size <= 1) &&
      // What the engine can read in the structure column. Filtering here rather than
      // letting the request be refused is what keeps the baseline right: the form picks
      // its mandatory baseline from this list, and with sequence and molecule baselines
      // both flagged it was choosing the molecule one for protein data and failing every
      // run. `undefined` leaves the list unfiltered, for a caller with no dataset yet.
      (structureKind === undefined ||
        engine.structure_kinds === undefined ||
        engine.structure_kinds.includes(structureKind)),
  );
}

/** Joint engines left out of `enginesForTargets` because the targets mix kinds. */
export function jointEnginesRefused(engines: Engine[], targets: HasKind[]): Engine[] {
  const mixed = new Set(targets.map((target) => target.kind)).size > 1;
  return mixed ? engines.filter((engine) => engine.supports_multitask) : [];
}

/** How an engine trains on several targets; nothing to say about one. */
export function trainingKind(engine: Engine, targetCount: number): string | null {
  if (targetCount < 2) return null;
  return engine.supports_multitask
    ? `Trains one joint model on all ${targetCount} targets.`
    : `Trains ${targetCount} separate models, one per target.`;
}

/**
 * Which "How it learns" figure explains each engine. Hardcoded engine
 * knowledge again, kept beside the other two maps so an engine change shows
 * it to the reviewer. An id missing here shows no figure, never a wrong one.
 */
export type EngineFigure =
  | "forest"
  | "boosting"
  | "gaussian-process"
  | "message-passing"
  | "attention";

export const ENGINE_FIGURE: Record<string, EngineFigure> = {
  "ecfp4-randomforest": "forest",
  "ecfp4-xgboost": "boosting",
  "ecfp4-lightgbm": "boosting",
  "descriptors-xgboost": "boosting",
  "tanimoto-gp": "gaussian-process",
  "chemprop-dmpnn": "message-passing",
  "molformer-xl": "attention",
};
