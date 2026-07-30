"use client";

import { API_V1, customInstance } from "@/shared/lib/api/custom-instance";
import { downloadFile } from "@/shared/lib/api/download";
import type {
  CollectionResponse,
  PaginatedResponseCollectionResponse,
} from "@/shared/lib/api/model";
import { showSuccess } from "@/shared/lib/toast";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { ExportFormat } from "../types";

export const COLLECTIONS_KEY = ["collections"];
export const COLLECTION_KEY = ["collection"];

export function useCollections() {
  return useQuery({
    queryKey: COLLECTIONS_KEY,
    queryFn: () =>
      customInstance<PaginatedResponseCollectionResponse>({
        url: `${API_V1}/collections`,
        method: "GET",
      }),
  });
}

export function useCollection(id: string | undefined) {
  return useQuery({
    queryKey: [...COLLECTION_KEY, id],
    queryFn: () =>
      customInstance<CollectionResponse>({ url: `${API_V1}/collections/${id}`, method: "GET" }),
    enabled: Boolean(id),
  });
}

export function useCreateCollection() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: { name: string; run_id: string; row_ids: number[] }) =>
      customInstance<CollectionResponse>({
        url: `${API_V1}/collections`,
        method: "POST",
        data,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: COLLECTIONS_KEY });
      showSuccess("Collection saved");
    },
  });
}

/** `Soluble candidates.csv`, not `collection.csv`. */
function exportFilename(name: string, format: ExportFormat): string {
  const slug = name
    .trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .toLowerCase();
  return `${slug || "collection"}.${format}`;
}

/**
 * Export goes through fetch rather than a link: the authz token travels as a
 * header, so a plain `<a href>` would arrive unauthenticated.
 *
 * The filename comes from the collection's own name rather than the server's
 * `Content-Disposition`. That header is set, but CORS does not expose it to
 * JavaScript unless the server lists it in `Access-Control-Expose-Headers`, so
 * reading it cross-origin always yielded null and every file landed as
 * `collection.csv`. Naming it after what the scientist called it is both
 * fixable here and the better answer.
 */
export function useExportCollection() {
  return useMutation({
    mutationFn: ({
      id,
      name,
      format,
    }: {
      id: string;
      name: string;
      format: ExportFormat;
    }) =>
      downloadFile({
        url: `${API_V1}/collections/${id}/export?format=${format}`,
        filename: exportFilename(name, format),
      }),
  });
}
