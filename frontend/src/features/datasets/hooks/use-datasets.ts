"use client";

import {
  API_V1,
  type ApiError,
  customInstance,
  getApiBaseUrl,
  getAuthHeaders,
} from "@/shared/lib/api/custom-instance";
import type {
  CompoundPageResponse,
  DatasetBuildResponse,
  DatasetColumnsResponse,
  DatasetProfileResponse,
  DatasetResponse,
  PaginatedResponseDatasetResponse,
  PaginatedResponseProtocolResponse,
  ProfileComputingResponse,
  UploadResponse,
} from "@/shared/lib/api/model";
import { showSuccess } from "@/shared/lib/toast";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Dataset } from "../types";
import {
  DATASETS_KEY,
  DATASET_BUILD_KEY,
  DATASET_COMPOUNDS_KEY,
  DATASET_KEY,
  DATASET_PROFILE_KEY,
} from "./query-keys";

/**
 * A picker passes `limit: 200`, the server's cap, to see past the default page of 50.
 * ponytail: a picker sees the newest 200; past that it needs search, not a bigger page.
 */
export function useDatasets(cursor?: string, limit?: number, filters?: { folderId?: string }) {
  return useQuery({
    queryKey: [...DATASETS_KEY, cursor ?? null, limit ?? null, filters ?? null],
    queryFn: () =>
      customInstance<PaginatedResponseDatasetResponse>({
        url: `${API_V1}/datasets`,
        method: "GET",
        // URLSearchParams percent-encodes, which is what keeps an opaque
        // base64 cursor intact through the round trip.
        params: { cursor, limit, folder_id: filters?.folderId },
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
 *
 * Silent to the global toast: the wizard reports a failed upload itself.
 */
export function useUploadDatasetFile() {
  return useMutation({
    meta: { silent: true },
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
  targets: { column: string; kind: string; unit?: string | null; direction?: string | null }[];
  split: { strategy: string; seed: number };
  id_column?: string | null;
}

/**
 * Start building a Dataset in the background. Building runs RDKit over every row,
 * which takes minutes on a large file, so the server answers at once and the
 * caller follows the build with `useDatasetBuild`.
 *
 * Silent to the global toast, like every create in this feature: the caller
 * decides what a failure looks like.
 */
export function useStartDatasetBuild() {
  return useMutation<DatasetBuildResponse, ApiError, CreateDatasetInput>({
    meta: { silent: true },
    mutationFn: (data) =>
      customInstance<DatasetBuildResponse>({
        url: `${API_V1}/datasets/builds`,
        method: "POST",
        data,
      }),
  });
}

const BUILD_POLL_MS = 1000;

/**
 * A dataset build, polled every second until it succeeds or fails. On success the
 * dataset list is refreshed, so the new dataset is there when the caller navigates.
 */
export function useDatasetBuild(id: string | null) {
  const queryClient = useQueryClient();
  return useQuery({
    queryKey: [...DATASET_BUILD_KEY, id],
    queryFn: async ({ signal }) => {
      const build = await customInstance<DatasetBuildResponse>({
        url: `${API_V1}/datasets/builds/${id}`,
        method: "GET",
        signal,
      });
      if (build.status === "succeeded") {
        await queryClient.invalidateQueries({ queryKey: DATASETS_KEY });
      }
      return build;
    },
    enabled: id !== null,
    refetchInterval: (query) => (query.state.data?.status === "running" ? BUILD_POLL_MS : false),
  });
}

/**
 * The Dataset's profile -- what it is made of.
 *
 * `staleTime: Infinity`: a Dataset is immutable and content-addressed, so its
 * profile is a pure function of data that cannot change. The backend caches it
 * beside the snapshot for the same reason; there is nothing to revalidate.
 *
 * `retry: false`: the first request computes the profile and can take seconds
 * on a large file. Retrying a timeout would start a second identical RDKit pass
 * rather than wait for the first, which makes a slow page slower.
 */
/** Snapshot columns that may be named as the identifier. */
export function useDatasetColumns(id: string, enabled: boolean) {
  return useQuery({
    queryKey: [...DATASET_KEY, id, "columns"],
    queryFn: () =>
      customInstance<DatasetColumnsResponse>({
        url: `${API_V1}/datasets/${id}/columns`,
        method: "GET",
      }),
    enabled,
    staleTime: Number.POSITIVE_INFINITY,
  });
}

export function useSetDatasetIdColumn() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, idColumn }: { id: string; idColumn: string | null }) =>
      customInstance<DatasetResponse>({
        url: `${API_V1}/datasets/${id}/id-column`,
        method: "PUT",
        data: { id_column: idColumn },
      }),
    onSuccess: (dataset) => {
      queryClient.setQueryData([...DATASET_KEY, dataset.id], dataset);
      queryClient.invalidateQueries({ queryKey: [...DATASET_COMPOUNDS_KEY, dataset.id] });
      // Scorecards and map tooltips of its protocols show these IDs too: mark
      // everything stale, refetching nothing now.
      queryClient.invalidateQueries({ refetchType: "none" });
    },
  });
}

/**
 * The protocols trained on a dataset: what stands between it and deletion.
 * Keyed under "protocols" (the protocols feature's root key, not imported to
 * keep the two features' barrels from importing each other) so anything that
 * invalidates protocols refreshes this too.
 */
export function useDatasetProtocols(datasetId: string, enabled: boolean) {
  return useQuery({
    queryKey: ["protocols", "by-dataset", datasetId],
    queryFn: () =>
      customInstance<PaginatedResponseProtocolResponse>({
        url: `${API_V1}/protocols`,
        method: "GET",
        params: { dataset_id: datasetId, limit: 200 },
      }),
    enabled,
  });
}

export function useDeleteDataset() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      customInstance<void>({ url: `${API_V1}/datasets/${id}`, method: "DELETE" }),
    // The dialog shows the error; no second toast.
    meta: { silent: true },
    onSuccess: () => {
      showSuccess("Dataset deleted.");
      // Mark everything stale without refetching: the page being left would
      // otherwise refetch the dataset it just deleted and flash a 404.
      queryClient.invalidateQueries({ refetchType: "none" });
    },
  });
}

/** How often to ask again while the server is still computing a profile. */
export const PROFILE_POLL_MS = 3_000;

/** The server's 202 answer: the profile is being computed, since `started_at`. */
export function isComputing(
  data: DatasetProfileResponse | ProfileComputingResponse | undefined,
): data is ProfileComputingResponse {
  return (data as ProfileComputingResponse | undefined)?.status === "computing";
}

export function profileRefetchInterval(
  data: DatasetProfileResponse | ProfileComputingResponse | undefined,
): number | false {
  return isComputing(data) ? PROFILE_POLL_MS : false;
}

/**
 * The profile, or a 202 saying it is being computed. Computing a large one takes
 * minutes, in the background on the server; asking again joins that computation
 * rather than starting another, so polling (and reloading) is safe. Once saved it
 * never changes, so a ready profile is never refetched.
 */
export function useDatasetProfile(id: string | undefined, target = 0) {
  return useQuery({
    queryKey: [...DATASET_PROFILE_KEY, id, target],
    queryFn: () =>
      customInstance<DatasetProfileResponse | ProfileComputingResponse>({
        url: `${API_V1}/datasets/${id}/profile`,
        method: "GET",
        params: { target },
      }),
    enabled: Boolean(id),
    staleTime: (query) => (isComputing(query.state.data) ? 0 : Number.POSITIVE_INFINITY),
    refetchInterval: (query) => profileRefetchInterval(query.state.data),
    retry: false,
  });
}

export interface CompoundQuery extends Record<string, unknown> {
  offset?: number;
  limit?: number;
  sort?: "target" | "split";
  /** Which target `sort: "target"` orders by, as an index into the dataset's targets. */
  target?: number;
  sort_dir?: "asc" | "desc";
  split?: "train" | "validation" | "test";
  /** Case-insensitive "contains" on the compound's ID. */
  q?: string;
}

/** A page of the frozen snapshot's own rows. Immutable, so the same page is
 *  the same rows forever -- hence `staleTime: Infinity` here too. */
export function useDatasetCompounds(id: string | undefined, query: CompoundQuery) {
  return useQuery({
    queryKey: [...DATASET_COMPOUNDS_KEY, id, query],
    queryFn: () =>
      customInstance<CompoundPageResponse>({
        url: `${API_V1}/datasets/${id}/compounds`,
        method: "GET",
        params: query,
      }),
    enabled: Boolean(id),
    staleTime: Number.POSITIVE_INFINITY,
    placeholderData: (previous) => previous,
  });
}
