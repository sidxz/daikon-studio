import { create } from "zustand";
import { persist } from "zustand/middleware";

// Must match the key the anti-flash script in app/layout.tsx reads before hydration.
const STORAGE_KEY = "ds-font";

export type FontFamily = "plex" | "inter" | "merriweather";

interface FontFamilyState {
  font: FontFamily;
  setFont: (font: FontFamily) => void;
}

export const useFontFamilyStore = create<FontFamilyState>()(
  persist(
    (set) => ({
      font: "inter",
      setFont: (font) => set({ font }),
    }),
    { name: STORAGE_KEY },
  ),
);
