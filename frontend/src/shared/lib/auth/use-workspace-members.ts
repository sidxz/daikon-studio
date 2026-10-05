"use client";

import { useQuery } from "@tanstack/react-query";
import { useCallback } from "react";
import { getDuarClient } from "./config";

/**
 * Everyone in the current workspace, from Duar, which owns identity. Studio stores
 * only user ids (a run's `requested_by`), so names are always current.
 */
export function useWorkspaceMembers() {
  return useQuery({
    queryKey: ["duar", "members"],
    queryFn: () => getDuarClient().listMembers(),
    staleTime: 10 * 60_000,
    retry: false,
  });
}

/**
 * A member's name by user id. Undefined while loading, when Duar cannot be reached,
 * or for someone no longer in the workspace -- callers then show no name at all.
 */
export function useMemberName(): (userId: string | null | undefined) => string | undefined {
  const { data } = useWorkspaceMembers();
  return useCallback(
    (userId) =>
      userId ? data?.find((member) => member.user_id === userId)?.name || undefined : undefined,
    [data],
  );
}
