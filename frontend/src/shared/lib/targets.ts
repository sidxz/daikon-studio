import type { ReadoutResponse } from "@/shared/lib/api/model";

/**
 * The target columns a Protocol predicts, recovered from its readouts. Mirrors
 * the backend's `target_columns_of`: every target has exactly one readout named
 * after its own column, and only a binary one adds a second, probability, one.
 */
export function targetsOf(readouts: ReadoutResponse[]): string[] {
  return readouts
    .filter((readout) => readout.type !== "probability")
    .map((readout) => readout.name);
}

/**
 * The results column holding one target's uncertainty, which is also the name
 * the results endpoint sorts and filters by. Mirrors the backend's
 * `uncertainty_column`: plain `uncertainty` for a one-target Protocol.
 */
export function uncertaintyColumn(target: string, targetCount: number): string {
  return targetCount === 1 ? "uncertainty" : `${target}_uncertainty`;
}
