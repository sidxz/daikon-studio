import type { PageOwnerKind } from "./types";

export const pagesKeys = {
  all: ["pages"] as const,
  owner: (kind: PageOwnerKind, id: string) => ["pages", "owner", kind, id] as const,
  detail: (id: string) => ["pages", id] as const,
  content: (sha: string) => ["pages", "content", sha] as const,
  revisions: (id: string) => ["pages", id, "revisions"] as const,
};
