/**
 * Staleness tiers. The QueryClient default is DEFAULT, so a hook only names one
 * when it wants a different tier -- restating the default is a no-op.
 */
export const STALE_TIME = {
  SHORT: 30_000,
  DEFAULT: 60_000,
  MEDIUM: 5 * 60_000,
  LONG: 30 * 60_000,
  STATIC: Number.POSITIVE_INFINITY,
} as const;

/**
 * A chemical-space map never changes once drawn, but training draws it last and
 * links the protocol before it finishes, so "missing" can turn "ready" a minute
 * later. Keep a missing answer fresh for nothing and a ready one for long.
 */
export function mapStaleTime(query: { state: { data?: { status?: string } } }): number {
  return query.state.data?.status === "missing" ? 0 : STALE_TIME.LONG;
}

/** Poll interval for a Run, per the design: 202-then-poll, no websockets. */
export const RUN_POLL_MS = 2_000;

const TERMINAL = new Set(["ready", "failed", "cancelled"]);

/** A Run in a terminal status changes again only if someone retries it. */
export function isTerminal(status: string | undefined): boolean {
  return status !== undefined && TERMINAL.has(status);
}

/** Poll interval once a refetch has failed with data still on screen: a laptop
 *  waking before its Wi-Fi, or the API mid-restart, must not end the watch. */
export const RUN_RETRY_POLL_MS = 10_000;

/**
 * `refetchInterval` for a query that returns one Run: poll until it is
 * terminal. A request that fails with *nothing* loaded stops the poll -- a 404
 * refetched every two seconds is a screen stuck on its skeleton. A request
 * that fails with data already on screen is a blip, and the poll backs off
 * instead of stopping. It lives here, not in the runs feature, because
 * protocols polls Runs too and runs imports protocols.
 */
export function pollInterval(query: {
  state: { status: string; data?: { status?: string } };
}): number | false {
  if (query.state.status === "error") {
    return query.state.data === undefined ? false : RUN_RETRY_POLL_MS;
  }
  return isTerminal(query.state.data?.status) ? false : RUN_POLL_MS;
}

/** Poll interval for the Runners list, so an online dot moves without a reload. */
export const RUNNER_POLL_MS = 10_000;
