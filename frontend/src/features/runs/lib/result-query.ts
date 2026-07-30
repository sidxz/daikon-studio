import type { SortModelItem } from "ag-grid-community";

/** The applicability below which a compound is extrapolation, not prediction. */
export const IN_DOMAIN_FLOOR = 0.5;

export interface ResultParams {
  sort_by?: string;
  sort_dir?: "asc" | "desc";
  filters?: string;
}

interface NumberFilter {
  filterType?: string;
  type?: string;
  filter?: number | null;
  filterTo?: number | null;
}

interface Bounds {
  min?: number;
  max?: number;
}

function boundsFor(model: NumberFilter): Bounds | null {
  const { type, filter, filterTo } = model;
  if (type === "greaterThanOrEqual" && filter != null) return { min: filter };
  if (type === "lessThanOrEqual" && filter != null) return { max: filter };
  if (type === "inRange" && filter != null && filterTo != null)
    return { min: filter, max: filterTo };
  // Any other operation is one the columns do not offer and the API cannot
  // honour; sending it would filter server-side by something else entirely.
  return null;
}

/**
 * AG Grid's sort and filter models, plus the in-domain switch, as query
 * parameters for `GET /runs/{id}/results`.
 *
 * Pure and separately tested because it is the one place where a client-side
 * control becomes a server-side promise: a mistranslation here shows a
 * chemist a filtered grid that was filtered by something else.
 */
export function buildResultParams({
  sortModel,
  filterModel,
  inDomainOnly,
}: {
  sortModel: SortModelItem[];
  filterModel: Record<string, NumberFilter>;
  inDomainOnly: boolean;
}): ResultParams {
  const params: ResultParams = {};

  // Single-column sort: the API sorts by one column, and pretending otherwise
  // would silently drop the rest of the user's intent.
  const [primary] = sortModel;
  if (primary) {
    params.sort_by = primary.colId;
    params.sort_dir = primary.sort === "desc" ? "desc" : "asc";
  }

  const filters: Record<string, Bounds> = {};
  for (const [column, model] of Object.entries(filterModel ?? {})) {
    const bounds = boundsFor(model);
    if (bounds) filters[column] = bounds;
  }

  if (inDomainOnly) {
    // Both constraints hold at once, so the bounds intersect: the tighter
    // floor wins, and the user's own ceiling survives.
    const existing = filters.applicability ?? {};
    filters.applicability = {
      ...existing,
      min: Math.max(existing.min ?? IN_DOMAIN_FLOOR, IN_DOMAIN_FLOOR),
    };
  }

  if (Object.keys(filters).length > 0) params.filters = JSON.stringify(filters);
  return params;
}
