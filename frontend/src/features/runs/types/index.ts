import type { PredictionResponse, RunResponse } from "@/shared/lib/api/model";

export type Run = RunResponse;
export type Prediction = PredictionResponse;

/**
 * A prediction plus the offset it arrived at.
 *
 * `PredictionResponse` carries no id, but `POST /collections` wants `row_ids`
 * as plain integer offsets into this run's paging order. The datasource stamps
 * that offset on each row as the block arrives; it is also what lets AG Grid
 * keep a selection alive across blocks it has since discarded.
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
