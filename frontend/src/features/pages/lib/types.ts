import type { PageResponse, RevisionResponse } from "@/shared/lib/api/model";

export type PMDoc = { type: "doc"; attrs?: Record<string, unknown>; content?: unknown[] };

export type PageOwnerKind = "dataset" | "protocol" | "run";

export type PageView = PageResponse;
export type RevisionView = RevisionResponse;
