// Fetch wrapper backing the generated API client and every hand-written hook.
// Base URL is resolved at runtime from /api/config (see AuthProvider), never
// baked at build time. Auth headers come from the shared Duar singleton.

import { getDuarClient } from "@/shared/lib/auth/config";

let _baseUrl = "http://localhost:8002";

/**
 * Versioned API prefix, for hand-written hooks that build a URL themselves.
 * Compose as `${API_V1}/...` rather than repeating the literal, so a version
 * bump is one edit. (Generated clients embed the prefix already.)
 */
export const API_V1 = "/api/v1";

/**
 * Thrown for any non-2xx response.
 *
 * Extends `Error` so callers that only read `.message` keep working, and
 * retains the parsed `body` so callers that need the server's structured
 * payload can have it. That retention is load-bearing here: a 422 from
 * `POST /datasets` carries the entire ValidationReport, and that report -- not
 * the status code -- is the useful part of a rejection.
 */
export class ApiError extends Error {
  readonly status: number;
  /** Parsed JSON body, or `undefined` when empty or not JSON. Narrow before use. */
  readonly body: unknown;
  /** True for a 401 that a session renewal is already handling: nothing to toast. */
  readonly silent: boolean;

  constructor(message: string, status: number, body: unknown, silent = false) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
    this.silent = silent;
  }
}

let _onUnauthorized: (() => boolean) | null = null;
let _unauthorizedNotified = false;

/**
 * Registered by the dashboard layout; fired once per expiry, not once per failed
 * query. The handler returns whether it actually started a renewal: when it
 * declined (hidden tab, loop guard) the next 401 may ask again.
 */
export function setUnauthorizedHandler(handler: (() => boolean) | null): void {
  _onUnauthorized = handler;
  _unauthorizedNotified = false;
}

/** Called by AuthProvider once runtime config has loaded. */
export function setApiBaseUrl(url: string): void {
  _baseUrl = url.replace(/\/$/, "");
}

/** The current base URL, for direct fetches that bypass this wrapper (uploads, downloads). */
export function getApiBaseUrl(): string {
  return _baseUrl;
}

/**
 * Duar auth headers for direct `fetch` calls. Returns `{}` on the server or
 * when unauthenticated, so the same guard applies everywhere instead of being
 * re-derived per call site.
 */
export function getAuthHeaders(): Record<string, string> {
  const client = typeof window !== "undefined" ? getDuarClient() : null;
  return client?.isAuthenticated ? client.getHeaders() : {};
}

export const customInstance = async <T>({
  url,
  method,
  params,
  data,
  headers,
  signal,
}: {
  url: string;
  method: "GET" | "POST" | "PUT" | "DELETE" | "PATCH";
  params?: Record<string, unknown>;
  data?: unknown;
  headers?: Record<string, string>;
  signal?: AbortSignal;
}): Promise<T> => {
  // URLSearchParams percent-encodes for us, which is what keeps an opaque
  // base64 cursor intact: a raw `+` in a query string decodes to a space
  // server-side, and that exact bug caused an infinite pagination loop in a
  // sibling app. Arrays become repeated keys, matching FastAPI's `list[T]`.
  const searchParams = new URLSearchParams();
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value == null) continue;
      if (Array.isArray(value)) {
        for (const item of value) {
          if (item == null) continue;
          searchParams.append(key, String(item));
        }
      } else {
        searchParams.append(key, String(value));
      }
    }
  }
  const query = searchParams.toString() ? `?${searchParams.toString()}` : "";

  const isFormData = typeof FormData !== "undefined" && data instanceof FormData;

  const response = await fetch(`${_baseUrl}${url}${query}`, {
    method,
    headers: {
      ...(isFormData ? {} : { "Content-Type": "application/json" }),
      ...getAuthHeaders(),
      ...headers,
    },
    signal,
    ...(data ? { body: isFormData ? (data as FormData) : JSON.stringify(data) } : {}),
  });

  if (!response.ok) {
    // Three shapes reach here: the app's domain errors `{error, message,
    // detail?}` (error_handlers.py), FastAPI's `{detail: "..."}`, and request
    // validation's `{detail: [{loc, msg}]}`. Flatten each into a readable
    // message, and keep the whole body on the error. The domain shape's
    // `message` is written for a person ("Requires admin role or higher"), so
    // it is shown as-is; reading only `detail` turned every one of them into a
    // bare "API error: 403".
    let body: unknown;
    let detail: string | undefined;
    let message: string | undefined;
    try {
      body = await response.json();
      const parsed = body as { detail?: unknown; message?: unknown } | null;
      if (typeof parsed?.message === "string") {
        message =
          typeof parsed.detail === "string" ? `${parsed.message} (${parsed.detail})` : parsed.message;
      } else if (typeof parsed?.detail === "string") {
        detail = parsed.detail;
      } else if (Array.isArray(parsed?.detail)) {
        detail = parsed.detail
          .map((entry: { loc?: unknown; msg?: unknown }) => {
            const loc = Array.isArray(entry.loc) ? entry.loc.join(".") : "";
            const msg = typeof entry.msg === "string" ? entry.msg : JSON.stringify(entry);
            return loc ? `${loc}: ${msg}` : msg;
          })
          .join("; ");
      }
    } catch {
      // Body was empty or not JSON -- fall through with no detail.
    }
    if (response.status === 401 && _onUnauthorized) {
      if (!_unauthorizedNotified) {
        // A handler that returns nothing is taken to have started the renewal.
        _unauthorizedNotified = _onUnauthorized() !== false;
      }
      throw new ApiError("Your session expired; signing you back in", 401, body, true);
    }
    throw new ApiError(
      message ??
        (detail ? `API error: ${response.status} — ${detail}` : `API error: ${response.status}`),
      response.status,
      body,
    );
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
};
