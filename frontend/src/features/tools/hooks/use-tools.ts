"use client";

import { API_V1, customInstance } from "@/shared/lib/api/custom-instance";
import { STALE_TIME } from "@/shared/lib/query-defaults";
import { useQuery } from "@tanstack/react-query";
import type { Tool } from "../types";
import { TOOLS_KEY } from "./query-keys";

/**
 * The tool catalogue. A module-level list in the backend, versioned with it, so
 * it only changes on deploy -- hence the long staleness rather than the default
 * tier, same as the engine catalogue.
 */
export function useTools() {
  return useQuery({
    queryKey: TOOLS_KEY,
    queryFn: () => customInstance<Tool[]>({ url: `${API_V1}/tools`, method: "GET" }),
    staleTime: STALE_TIME.LONG,
  });
}
