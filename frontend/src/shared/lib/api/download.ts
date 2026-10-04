import { getApiBaseUrl, getAuthHeaders } from "./custom-instance";

/** Trigger a browser "save as" for an in-memory Blob. */
export function saveBlob(blob: Blob, filename: string): void {
  const blobUrl = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = blobUrl;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  // Not synchronously: some browsers (Firefox, older Safari) start reading the
  // blob after click() returns, and an immediate revoke cancels the download.
  setTimeout(() => URL.revokeObjectURL(blobUrl), 30_000);
}

/** `Soluble candidates` -> `soluble-candidates.csv`, so a file is named for what the scientist called it. */
export function fileName(name: string, extension: string, fallback: string): string {
  const slug = name
    .trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-{2,}/g, "-")
    .toLowerCase();
  return `${slug || fallback}.${extension}`;
}

/** Save in-memory text -- how every "Download template" button works. */
export function saveText(text: string, filename: string, mime = "text/csv"): void {
  saveBlob(new Blob([text], { type: mime }), filename);
}

function filenameFromContentDisposition(disposition: string | null): string | undefined {
  if (!disposition) return undefined;
  return disposition.match(/filename="?([^"]+)"?/)?.[1];
}

/**
 * Download from an authenticated endpoint.
 *
 * A plain `<a href>` cannot do this: the authz token lives in localStorage and
 * travels as a header, so the request has to go through fetch. The server's
 * own `Content-Disposition` names the file.
 */
export async function downloadFile({
  url,
  method = "GET",
  data,
  filename,
  fallbackFilename = "download",
}: {
  url: string;
  method?: "GET" | "POST";
  data?: unknown;
  filename?: string;
  fallbackFilename?: string;
}): Promise<void> {
  const headers: Record<string, string> = { ...getAuthHeaders() };
  if (data !== undefined) headers["Content-Type"] = "application/json";

  const response = await fetch(`${getApiBaseUrl()}${url}`, {
    method,
    headers,
    ...(data !== undefined ? { body: JSON.stringify(data) } : {}),
  });

  if (!response.ok) {
    // The domain error's own words ("an Excel sheet holds at most ..."), not a status code.
    const body = (await response.json().catch(() => null)) as {
      message?: unknown;
      detail?: unknown;
    } | null;
    const message = typeof body?.message === "string" ? body.message : null;
    const detail = typeof body?.detail === "string" ? ` (${body.detail})` : "";
    throw new Error(message ? `${message}${detail}` : `Download failed (${response.status})`);
  }

  saveBlob(
    await response.blob(),
    filename ??
      filenameFromContentDisposition(response.headers.get("content-disposition")) ??
      fallbackFilename,
  );
}
