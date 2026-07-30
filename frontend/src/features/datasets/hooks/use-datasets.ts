"use client";

import {
  API_V1,
  type ApiError,
  customInstance,
  getApiBaseUrl,
  getAuthHeaders,
} from "@/shared/lib/api/custom-instance";
import type {
  DatasetResponse,
  PaginatedResponseDatasetResponse,
  UploadResponse,
} from "@/shared/lib/api/model";
import { showSuccess } from "@/shared/lib/toast";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Dataset } from "../types";
import { DATASETS_KEY, DATASET_KEY } from "./query-keys";

export function useDatasets(cursor?: string) {
  return useQuery({
    queryKey: [...DATASETS_KEY, cursor ?? null],
    queryFn: () =>
      customInstance<PaginatedResponseDatasetResponse>({
        url: `${API_V1}/datasets`,
        method: "GET",
        // URLSearchParams percent-encodes, which is what keeps an opaque
        // base64 cursor intact through the round trip.
        params: cursor ? { cursor } : undefined,
      }),
  });
}

export function useDataset(id: string | undefined) {
  return useQuery({
    queryKey: [...DATASET_KEY, id],
    queryFn: () =>
      customInstance<DatasetResponse>({ url: `${API_V1}/datasets/${id}`, method: "GET" }),
    enabled: Boolean(id),
  });
}

/**
 * Park the raw file and get back an opaque ref.
 *
 * Deliberately a bare `fetch` rather than `customInstance`: the body is
 * multipart, and letting the browser set its own boundary is the whole point --
 * a hand-set Content-Type here produces a boundary mismatch and a 422 that
 * reads like a validation failure.
 */
export function useUploadDatasetFile() {
  return useMutation({
    mutationFn: async (file: File): Promise<string> => {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch(`${getApiBaseUrl()}${API_V1}/datasets/uploads`, {
        method: "POST",
        headers: getAuthHeaders(),
        body,
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(
          typeof detail?.detail === "string" ? detail.detail : `Upload failed (${response.status})`,
        );
      }
      return ((await response.json()) as UploadResponse).upload_ref;
    },
  });
}

export interface CreateDatasetInput {
  name: string;
  upload_ref: string;
  structure_column: string;
  target: { column: string; kind: string; unit?: string | null; direction?: string | null };
  split: { strategy: string; seed: number };
}

/**
 * Freeze a Dataset.
 *
 * No `onError` toast: a 422 here carries the entire ValidationReport, and that
 * report is the useful part of the rejection. The caller renders it as a page.
 * Collapsing it into a toast would throw away exactly the information the
 * scientist needs to fix their file.
 */
export function useCreateDataset() {
  const queryClient = useQueryClient();
  return useMutation<Dataset, ApiError, CreateDatasetInput>({
    mutationFn: (data) =>
      customInstance<Dataset>({ url: `${API_V1}/datasets`, method: "POST", data }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: DATASETS_KEY });
      showSuccess("Dataset frozen");
    },
  });
}
