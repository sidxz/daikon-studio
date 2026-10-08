import { API_V1, customInstance } from "@/shared/lib/api/custom-instance";

import type { PMDoc, PageOwnerKind, PageView, RevisionView } from "./types";

const base = `${API_V1}/pages`;

// Mirrors backend interface/routes/pages.py. Every mutation answers with the fresh
// PageView, but callers still invalidate: the cache is the source of `version`.
export const pagesApi = {
  create: (title: string, ownerKind: PageOwnerKind, ownerId: string) =>
    customInstance<PageView>({
      url: base,
      method: "POST",
      data: { title, owner_kind: ownerKind, owner_id: ownerId },
    }),
  get: (id: string) => customInstance<PageView>({ url: `${base}/${id}`, method: "GET" }),
  revise: (id: string, body: PMDoc, expectedVersion: number) =>
    customInstance<PageView>({
      url: `${base}/${id}`,
      method: "PATCH",
      data: { body, expected_version: expectedVersion },
    }),
  retitle: (id: string, title: string) =>
    customInstance<PageView>({ url: `${base}/${id}/title`, method: "PUT", data: { title } }),
  revisions: (id: string) =>
    customInstance<RevisionView[]>({ url: `${base}/${id}/revisions`, method: "GET" }),
  list: (ownerKind: PageOwnerKind, ownerId: string, archived = false) =>
    customInstance<PageView[]>({
      url: base,
      method: "GET",
      params: { owner_kind: ownerKind, owner_id: ownerId, archived },
    }),
  archive: (id: string, restore = false) =>
    customInstance<PageView>({ url: `${base}/${id}/archive`, method: "POST", data: { restore } }),
  remove: (id: string) => customInstance<void>({ url: `${base}/${id}`, method: "DELETE" }),
  getContent: (id: string, sha256: string) =>
    customInstance<PMDoc>({ url: `${base}/${id}/content/${sha256}`, method: "GET" }),
};
