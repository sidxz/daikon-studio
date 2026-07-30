import { create } from "zustand";
import { persist } from "zustand/middleware";

// Must match the key the anti-flash script in app/layout.tsx reads before hydration.
const STORAGE_KEY = "ds-font-scale";

// Percent of the browser's default font-size. Relative rather than absolute px
// so a user who raised their browser font for accessibility keeps that
// baseline; this scales on top of it.
export const FONT_SCALE_MIN = 80;
export const FONT_SCALE_MAX = 120;
export const FONT_SCALE_STEP = 5;
export const FONT_SCALE_DEFAULT = 100;

const clamp = (value: number) => Math.min(FONT_SCALE_MAX, Math.max(FONT_SCALE_MIN, value));

interface FontScaleState {
  scale: number;
  setScale: (scale: number) => void;
  reset: () => void;
}

export const useFontScaleStore = create<FontScaleState>()(
  persist(
    (set) => ({
      scale: FONT_SCALE_DEFAULT,
      setScale: (scale) => set({ scale: clamp(scale) }),
      reset: () => set({ scale: FONT_SCALE_DEFAULT }),
    }),
    {
      name: STORAGE_KEY,
      // Re-clamp on rehydrate: a hand-edited or stale localStorage value must
      // not be able to set an unreadable root font size.
      merge: (persisted, current) => {
        const scale = (persisted as Partial<FontScaleState> | undefined)?.scale;
        return {
          ...current,
          scale: typeof scale === "number" ? clamp(scale) : current.scale,
        };
      },
    },
  ),
);
