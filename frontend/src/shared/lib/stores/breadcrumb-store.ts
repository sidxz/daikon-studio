import { useEffect } from "react";
import { create } from "zustand";

/**
 * Breadcrumb labels a page declares for itself.
 *
 * Two mechanisms, because two problems. `useBreadcrumbOverride` maps one URL
 * segment to a human label; `useBreadcrumbTrail` replaces the whole trail when
 * the URL alone cannot describe where you are. Both exist so a UUID never
 * reaches the breadcrumb bar -- the URL-derived fallback would print one
 * verbatim, and a user thinks in names, not identifiers.
 *
 * Convention: app-wide stores live here in shared/lib/stores; a store local to
 * one feature lives in that feature's hooks/ as use-*.ts.
 */
export interface Crumb {
  label: string;
  href?: string;
}

interface BreadcrumbState {
  overrides: Map<string, string>;
  trail: Crumb[] | null;
  setOverride: (segment: string, label: string) => void;
  clearOverride: (segment: string) => void;
  setTrail: (trail: Crumb[] | null) => void;
}

const useBreadcrumbStore = create<BreadcrumbState>()((set) => ({
  overrides: new Map(),
  trail: null,
  setOverride: (segment, label) =>
    set((state) => {
      const overrides = new Map(state.overrides);
      overrides.set(segment, label);
      return { overrides };
    }),
  clearOverride: (segment) =>
    set((state) => {
      const overrides = new Map(state.overrides);
      overrides.delete(segment);
      return { overrides };
    }),
  setTrail: (trail) => set({ trail }),
}));

export function useBreadcrumbOverrides(): Map<string, string> {
  return useBreadcrumbStore((state) => state.overrides);
}

export function useBreadcrumbTrailValue(): Crumb[] | null {
  return useBreadcrumbStore((state) => state.trail);
}

/** Label one URL segment. Cleared on unmount. */
export function useBreadcrumbOverride(segment: string | undefined, label: string | undefined) {
  const setOverride = useBreadcrumbStore((state) => state.setOverride);
  const clearOverride = useBreadcrumbStore((state) => state.clearOverride);
  useEffect(() => {
    if (!segment || !label) return;
    setOverride(segment, label);
    return () => clearOverride(segment);
  }, [segment, label, setOverride, clearOverride]);
}

/** Declare the whole trail. Cleared on unmount. */
export function useBreadcrumbTrail(trail: Crumb[] | null) {
  const setTrail = useBreadcrumbStore((state) => state.setTrail);
  // Serialised so a fresh array literal on every render does not re-fire the
  // effect -- the same referential-stability trap that once turned a polling
  // hook into a request storm in a sibling app.
  const key = trail ? JSON.stringify(trail) : null;
  useEffect(() => {
    setTrail(key ? (JSON.parse(key) as Crumb[]) : null);
    return () => setTrail(null);
  }, [key, setTrail]);
}
