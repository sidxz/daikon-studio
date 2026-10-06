"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useRef } from "react";

/**
 * Filter state kept in the URL query, so a filtered view can be linked and survives
 * a reload. `set` merges: a value of `undefined` or "" removes the key.
 */
export function useUrlParams() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const query = params.toString();
  const navigation = useRef({ pathname, observed: query, query, pending: [] as string[] });
  const state = navigation.current;
  if (state.pathname !== pathname) {
    Object.assign(state, { pathname, observed: query, query, pending: [] });
  } else if (state.observed !== query) {
    state.observed = query;
    const acknowledged = state.pending.indexOf(query);
    if (acknowledged >= 0) state.pending.splice(0, acknowledged + 1);
    else state.pending = [];
    state.query = state.pending.at(-1) ?? query;
  }
  const set = (updates: Record<string, string | undefined>) => {
    // Merge rapid filter changes even while the router is committing an earlier one.
    const next = new URLSearchParams(state.query);
    for (const [key, value] of Object.entries(updates)) {
      if (value) next.set(key, value);
      else next.delete(key);
    }
    const query = next.toString();
    if (query === state.query) return;
    state.query = query;
    state.pending.push(query);
    router.replace(query ? `${pathname}?${query}` : pathname);
  };
  return { params, set };
}
