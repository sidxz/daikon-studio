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

/** Poll interval for a Run, per the design: 202-then-poll, no websockets. */
export const RUN_POLL_MS = 2_000;

/** Poll interval for the Runners list, so an online dot moves without a reload. */
export const RUNNER_POLL_MS = 10_000;
