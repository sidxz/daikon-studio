"use client";

import { customInstance } from "@/shared/lib/api/custom-instance";
import { STALE_TIME } from "@/shared/lib/query-defaults";
import { useQuery } from "@tanstack/react-query";

interface ApiVersion {
  service: string;
  version: string;
}

/** Backend build identity, for the About card. Unauthenticated endpoint. */
export function useApiVersion(enabled = true) {
  return useQuery({
    queryKey: ["api-version"],
    queryFn: () => customInstance<ApiVersion>({ url: "/version", method: "GET" }),
    staleTime: STALE_TIME.STATIC,
    enabled,
  });
}
