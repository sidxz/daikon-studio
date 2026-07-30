"use client";

import { API_V1, customInstance } from "@/shared/lib/api/custom-instance";
import { STALE_TIME } from "@/shared/lib/query-defaults";
import { useQuery } from "@tanstack/react-query";
import type { Engine } from "../types";
import { ENGINES_KEY } from "./query-keys";

/**
 * The engine catalogue. In-tree and versioned with the backend, so it only
 * changes on deploy -- hence the long staleness rather than the default tier.
 */
export function useEngines() {
  return useQuery({
    queryKey: ENGINES_KEY,
    queryFn: () => customInstance<Engine[]>({ url: `${API_V1}/engines`, method: "GET" }),
    staleTime: STALE_TIME.LONG,
  });
}
