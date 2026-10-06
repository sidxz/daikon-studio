import type { SortModelItem } from "ag-grid-community";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { TriageRow } from "../types";
import type { NumberFilter } from "./result-query";

export interface ResultView {
  filters: Record<string, NumberFilter>;
  sort: SortModelItem[];
  inDomainOnly: boolean;
  selected: Record<string, TriageRow>;
  firstRow: number;
}

export const EMPTY_VIEW: ResultView = {
  filters: {},
  sort: [],
  inDomainOnly: false,
  selected: {},
  firstRow: 0,
};

/** Non-integer keys retain selection order, including a ranked bulk selection. */
export const selectionKey = (rowId: number) => `row-${rowId}`;

// Keep working selections in memory only. Returning from a collection restores
// the run, without persisting compound data to localStorage.
export const useResultViews = create<{
  views: Record<string, ResultView>;
  update: (runId: string, patch: Partial<ResultView>) => void;
}>((set) => ({
  views: {},
  update: (runId, patch) =>
    set((state) => ({
      views: { ...state.views, [runId]: { ...(state.views[runId] ?? EMPTY_VIEW), ...patch } },
    })),
}));

export const useColumnChoices = create<{
  protocols: Record<string, Record<string, boolean>>;
  setShown: (protocol: string, choices: Record<string, boolean>) => void;
}>()(
  persist(
    (set) => ({
      protocols: {},
      setShown: (protocol, choices) =>
        set((state) => ({
          protocols: {
            ...state.protocols,
            [protocol]: { ...state.protocols[protocol], ...choices },
          },
        })),
    }),
    { name: "ds-triage-protocol-columns" },
  ),
);
