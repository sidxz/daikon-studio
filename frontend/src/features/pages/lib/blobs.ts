"use client";

import { useEffect, useState } from "react";

import {
  API_V1,
  customInstance,
  getApiBaseUrl,
  getAuthHeaders,
} from "@/shared/lib/api/custom-instance";

export type UploadedBlob = { sha256: string; mime: string; size: number };

/** Uploads a file to the workspace's content-addressed page blob store. */
export function uploadBlob(file: File): Promise<UploadedBlob> {
  const form = new FormData();
  form.append("file", file);
  return customInstance<UploadedBlob>({ url: `${API_V1}/pages/blobs`, method: "POST", data: form });
}

/**
 * Resolves a blob's sha256 to a renderable object URL.
 *
 * The blob route needs Duar auth headers (which a bare `<img src>` can't carry)
 * and answers with `Content-Disposition: attachment` (so a direct URL would
 * download, not render). Both are moot for `fetch`, so we fetch the bytes
 * ourselves and hand the `<img>` an object URL instead. A `cancelled` flag guards
 * state updates from a stale fetch, and the object URL is revoked on cleanup.
 */
export function useBlobObjectUrl(sha256: string | null): string | null {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!sha256) return;
    let cancelled = false;
    let objectUrl: string | null = null;

    void (async () => {
      try {
        const res = await fetch(`${getApiBaseUrl()}${API_V1}/pages/blobs/${sha256}`, {
          headers: getAuthHeaders(),
        });
        if (!res.ok || cancelled) return;
        const blob = await res.blob();
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      } catch {
        // ponytail: swallow — the NodeView just keeps showing its skeleton on a
        // failed fetch. Add a broken-image state if silent failures bite in practice.
      }
    })();

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [sha256]);

  return sha256 ? url : null;
}
