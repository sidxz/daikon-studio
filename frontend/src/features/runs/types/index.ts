import type { PredictionResponse, RunResponse } from "@/shared/lib/api/model";

export type Run = RunResponse;
export type Prediction = PredictionResponse;

/**
 * A prediction plus `row_id`, its position in the original results file --
 * minted server-side before any sort or filter, so it survives both.
 *
 * `POST /collections` wants exactly this as `row_ids`; it is also what lets
 * AG Grid keep a selection alive across blocks it has since discarded.
 */
export interface TriageRow extends PredictionResponse {
  __rowId: number;
}

export const RUN_STATUS_COPY: Record<string, string> = {
  pending: "Queued",
  running: "Running",
  ready: "Ready",
  failed: "Failed",
  cancelled: "Cancelled",
};
