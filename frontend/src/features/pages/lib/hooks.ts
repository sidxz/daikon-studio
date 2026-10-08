"use client";

import { type QueryClient, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { pagesApi } from "./api";
import { pagesKeys } from "./constants";
import type { PMDoc, PageOwnerKind, PageView } from "./types";

// Failed mutations toast globally (query-provider) unless `meta.silent`, so these
// hooks carry no onError of their own.

/**
 * The page's latest known server version, read from the cache at the moment of
 * the write rather than from a value captured in a render's closure. Every
 * successful write invalidates `pagesKeys.detail(id)`, so the cache — not React's
 * render timing — is the source of truth for "what version comes next". A genuine
 * concurrent edit still 409s; this only avoids *spurious* ones.
 */
export function currentVersion(qc: QueryClient, id: string): number {
  return qc.getQueryData<PageView>(pagesKeys.detail(id))?.version ?? 0;
}

export function usePage(id: string) {
  return useQuery({ queryKey: pagesKeys.detail(id), queryFn: () => pagesApi.get(id) });
}

export function useOwnerPages(kind: PageOwnerKind, ownerId: string, includeArchived = false) {
  return useQuery({
    queryKey: [...pagesKeys.owner(kind, ownerId), { archived: includeArchived }],
    queryFn: () => pagesApi.list(kind, ownerId, includeArchived),
  });
}

export function useRevisions(id: string) {
  return useQuery({
    queryKey: pagesKeys.revisions(id),
    queryFn: () => pagesApi.revisions(id),
  });
}

export function usePageContent(id: string, sha256: string | null) {
  return useQuery({
    queryKey: pagesKeys.content(sha256 ?? ""),
    queryFn: () => pagesApi.getContent(id, sha256 as string),
    enabled: !!sha256,
    staleTime: Number.POSITIVE_INFINITY, // content is immutable — a sha never changes
  });
}

export function useCreatePage(kind: PageOwnerKind, ownerId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (title: string) => pagesApi.create(title, kind, ownerId),
    onSuccess: () => qc.invalidateQueries({ queryKey: pagesKeys.owner(kind, ownerId) }),
  });
}

export function useRetitlePage(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (title: string) => pagesApi.retitle(id, title),
    onSuccess: () => qc.invalidateQueries({ queryKey: pagesKeys.all }),
  });
}

// Used for the one-shot revisions the editor doesn't drive: restoring an older
// version (page-history) and unlocking a page (page-container). The typing path
// calls pagesApi.revise directly through use-page-save. Silent — a 409 belongs to
// the caller (a conflict dialog or a specific message), not to a generic toast.
export function useRevisePage(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ body, expectedVersion }: { body: PMDoc; expectedVersion: number }) =>
      pagesApi.revise(id, body, expectedVersion),
    onSuccess: () => qc.invalidateQueries({ queryKey: pagesKeys.detail(id) }),
    meta: { silent: true },
  });
}

export function useArchivePage(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (restore: boolean) => pagesApi.archive(id, restore),
    onSuccess: (page) => {
      qc.setQueryData<PageView>(pagesKeys.detail(id), page);
      void qc.invalidateQueries({ queryKey: pagesKeys.all });
    },
  });
}

export function useDeletePage() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => pagesApi.remove(id),
    onSuccess: (_void, id) => {
      qc.removeQueries({ queryKey: pagesKeys.detail(id) });
      void qc.invalidateQueries({ queryKey: pagesKeys.all });
    },
  });
}
