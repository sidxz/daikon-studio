import type { ChemCellarImportResponse, ChemCellarRunResponse } from "@/shared/lib/api/model";

const STATUS: Record<string, string> = {
  draft: "Draft",
  in_progress: "In progress",
  completed: "Completed",
  approved: "Approved",
  rejected: "Rejected",
};

/**
 * A ChemCellar run date ("2026-06-05") as written. Read as UTC and shown in UTC: the
 * default `new Date("2026-06-05")` is UTC midnight, which is the previous day in every
 * US time zone.
 */
export function formatRunDate(isoDate: string): string {
  return new Date(`${isoDate}T00:00:00Z`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}

/** A run as the picker lists it. Selectable if it has readouts or plates. */
export function runOption(run: ChemCellarRunResponse): { label: string; disabled: boolean } {
  const parts = [formatRunDate(run.run_date), STATUS[run.status] ?? run.status];
  if (run.measured_count === 0 && run.plate_count === 0) {
    return { label: [...parts, "No compounds"].join(" · "), disabled: true };
  }
  parts.push(`${run.measured_count} compound${run.measured_count === 1 ? "" : "s"} measured`);
  parts.push(`${run.plate_count} plate${run.plate_count === 1 ? "" : "s"}`);
  return { label: parts.join(" · "), disabled: false };
}

/** The compounds the Predict button commits to: those of the tab in view, and only those. */
export function activeCompounds(
  tab: "csv" | "chemcellar",
  csvCount: number,
  imported: Pick<ChemCellarImportResponse, "compound_count"> | null,
): { count: number; ready: boolean } {
  const count = tab === "chemcellar" ? (imported?.compound_count ?? 0) : csvCount;
  return { count, ready: count > 0 };
}
