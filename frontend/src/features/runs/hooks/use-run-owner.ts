"use client";

import { useEffect, useState } from "react";

const KEY = "studio.runs.owner";

function remember(mine: boolean) {
  try {
    localStorage.setItem(KEY, mine ? "mine" : "all");
  } catch {
    // The URL still works when browser storage is unavailable.
  }
}

/** Explicit links take precedence; returning to /runs restores the last choice. */
export function useRunOwner(urlValue: string | null) {
  const [savedMine, setSavedMine] = useState<boolean | null>(null);
  const explicit = urlValue === "0" ? false : urlValue === "1" ? true : null;
  useEffect(() => {
    let value = true;
    try {
      value = localStorage.getItem(KEY) !== "all";
    } catch {
      // Keep the initial Mine preference when storage is unavailable.
    }
    if (explicit !== null) {
      value = explicit;
      remember(value);
    }
    setSavedMine(value);
  }, [explicit]);

  return {
    mine: explicit ?? savedMine ?? true,
    ready: explicit !== null || savedMine !== null,
    rememberOwner: (mine: boolean) => {
      remember(mine);
      setSavedMine(mine);
    },
  };
}
