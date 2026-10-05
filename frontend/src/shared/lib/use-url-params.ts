"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";

/**
 * Filter state kept in the URL query, so a filtered view can be linked and survives
 * a reload. `set` merges: a value of `undefined` or "" removes the key.
 */
export function useUrlParams() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const set = (updates: Record<string, string | undefined>) => {
    const next = new URLSearchParams(params.toString());
    for (const [key, value] of Object.entries(updates)) {
      if (value) next.set(key, value);
      else next.delete(key);
    }
    const query = next.toString();
    router.replace(query ? `${pathname}?${query}` : pathname);
  };
  return { params, set };
}
