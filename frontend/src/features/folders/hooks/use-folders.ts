"use client";

import { API_V1, type ApiError, customInstance } from "@/shared/lib/api/custom-instance";
import type { FolderKind, FolderListResponse, FolderResponse } from "@/shared/lib/api/model";
import { showSuccess } from "@/shared/lib/toast";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

export const FOLDERS_KEY = ["folders"];

// The item list roots, written out rather than imported: the datasets and protocols
// features import this one, and a barrel cycle would follow.
const ITEMS_KEY: Record<FolderKind, string[]> = { dataset: ["datasets"], protocol: ["protocols"] };
const ITEMS_PATH: Record<FolderKind, string> = { dataset: "datasets", protocol: "protocols" };

export function useFolders(kind: FolderKind) {
  return useQuery({
    queryKey: [...FOLDERS_KEY, kind],
    queryFn: () =>
      customInstance<FolderListResponse>({
        url: `${API_V1}/folders`,
        method: "GET",
        params: { kind },
      }),
  });
}

/** A folder change touches the folder counts and the filed items' lists. */
function useRefresh(kind: FolderKind) {
  const queryClient = useQueryClient();
  return () =>
    Promise.all([
      queryClient.invalidateQueries({ queryKey: [...FOLDERS_KEY, kind] }),
      queryClient.invalidateQueries({ queryKey: ITEMS_KEY[kind] }),
    ]);
}

// Create and rename are silent to the global toast: a duplicate name (409) shows
// inline in the name dialog.
export function useCreateFolder(kind: FolderKind) {
  const refresh = useRefresh(kind);
  return useMutation<FolderResponse, ApiError, string>({
    meta: { silent: true },
    mutationFn: (name) =>
      customInstance<FolderResponse>({
        url: `${API_V1}/folders`,
        method: "POST",
        data: { kind, name },
      }),
    onSuccess: async () => {
      await refresh();
      showSuccess("Folder created");
    },
  });
}

export function useRenameFolder(kind: FolderKind) {
  const refresh = useRefresh(kind);
  return useMutation<FolderResponse, ApiError, { id: string; name: string }>({
    meta: { silent: true },
    mutationFn: ({ id, name }) =>
      customInstance<FolderResponse>({
        url: `${API_V1}/folders/${id}`,
        method: "PATCH",
        data: { name },
      }),
    onSuccess: async () => {
      await refresh();
      showSuccess("Folder renamed");
    },
  });
}

export function useDeleteFolder(kind: FolderKind) {
  const refresh = useRefresh(kind);
  return useMutation<void, ApiError, string>({
    mutationFn: (id) => customInstance<void>({ url: `${API_V1}/folders/${id}`, method: "DELETE" }),
    onSuccess: async () => {
      await refresh();
      showSuccess("Folder deleted");
    },
  });
}

/** File an item into a folder, or out of any with `folderId: null`. */
export function useFileItem(kind: FolderKind) {
  const refresh = useRefresh(kind);
  const folders = useFolders(kind).data?.items;
  return useMutation<unknown, ApiError, { itemId: string; folderId: string | null }>({
    mutationFn: ({ itemId, folderId }) =>
      customInstance<unknown>({
        url: `${API_V1}/${ITEMS_PATH[kind]}/${itemId}/folder`,
        method: "PUT",
        data: { folder_id: folderId },
      }),
    onSuccess: async (_item, { folderId }) => {
      await refresh();
      const name = folders?.find((folder) => folder.id === folderId)?.name;
      showSuccess(folderId ? `Moved to ${name ?? "folder"}` : "Removed from folder");
    },
  });
}
