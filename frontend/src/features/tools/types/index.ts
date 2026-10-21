import type { SlotResponse, ToolManifestResponse } from "@/shared/lib/api/model";

// Generated DTOs, re-exported under domain names. Never redeclare their shape --
// `pnpm generate:api` is what keeps them true.
export type Tool = ToolManifestResponse;
export type Slot = SlotResponse;

/**
 * `SlotResponse.kind` is a bare `string` in the contract rather than a closed
 * enum, so codegen cannot give us a union. These are the kinds the backend's
 * `SlotKind` can emit, and the key order is the order they appear in a study.
 *
 * A kind is what flows between tools, so this table is also how a reader learns
 * the vocabulary a plan is written in -- keep the wording concrete.
 */
export const SLOT_LABELS: Record<string, string> = {
  compound_table: "Compounds",
  series: "Series",
  core_candidates: "Candidate cores",
  core: "Core",
  rgroup_table: "SAR table",
  pairs: "Matched pairs",
  effects: "Effect sizes",
  property_table: "Properties",
  correlation: "Correlation",
  alerts: "Alerts",
  model_fit: "Model fit",
  gaps: "Gaps",
  citations: "Citations",
};

/**
 * What a tool owes a person when it finishes, in plain words. `none` is
 * deliberately absent: a tool that needs nothing from anyone should say nothing
 * rather than wear a badge reading "nothing".
 */
export const SURFACE_COPY: Record<string, { title: string; detail: string }> = {
  view: {
    title: "Worth a look",
    detail: "Produces a result you read before going on, but nothing to change here.",
  },
  editor: {
    title: "Needs you",
    detail: "You pick or draw the answer. The tool proposes; it does not decide.",
  },
};

/**
 * The catalogue's tabs, in order. "All" first like the engines catalogue, then
 * the areas with SAR before the rest because that is the work Studio does today.
 * Key order here is tab order; an area the backend adds later and this table has
 * not learned about simply does not get a tab, which is better than an empty one.
 */
export const AREAS = [
  { value: "all", label: "All tools" },
  { value: "sar", label: "SAR analysis" },
  { value: "data", label: "Structures & properties" },
  { value: "structure", label: "Protein structure" },
  { value: "context", label: "What's known already" },
] as const;

export function slotLabel(slot: Slot): string {
  return SLOT_LABELS[slot.kind] ?? slot.kind.replace(/_/g, " ");
}
