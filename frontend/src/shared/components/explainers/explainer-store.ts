"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface ExplainerState {
  /** Figure ids this browser has closed. Absent means open, so every figure is open on first sight. */
  closed: Record<string, boolean>;
  setOpen: (id: string, open: boolean) => void;
}

// ponytail: persisted state can differ from the server render, but every
// explainer sits under data that loads client-side, so none renders on the
// server today. If one ever does, add skipHydration and rehydrate in an effect.
export const useExplainerStore = create<ExplainerState>()(
  persist(
    (set) => ({
      closed: {},
      setOpen: (id, open) =>
        set((state) => ({
          closed: open
            ? Object.fromEntries(Object.entries(state.closed).filter(([key]) => key !== id))
            : { ...state.closed, [id]: true },
        })),
    }),
    { name: "ds-explainers" },
  ),
);
