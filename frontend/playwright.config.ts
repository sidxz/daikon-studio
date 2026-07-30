import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end config.
 *
 * **Port 3103, not 3003.** The dev server you run by hand lives on 3003, and
 * localStorage is partitioned by origin -- which includes the port. Running the
 * suite on its own port means the crafted tokens in `auth.ts` land in a storage
 * bucket the studio you use never reads, so a test run can never leave your
 * local app in a strange state. It also means the suite does not fight
 * `make dev-fe` for a port, so both can run at once.
 *
 * **`APP_API_BASE_URL` points at a domain that does not resolve.** Every
 * `/api/v1/**` call is meant to be intercepted by `api-mock.ts`; if one ever
 * escapes, it fails DNS instead of reaching a real backend. That is deliberate:
 * these tests must never be able to talk to a live server, because the tokens
 * they carry are unsigned (see `auth.ts`).
 */
export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? "list" : [["list"], ["html", { open: "never" }]],

  use: {
    baseURL: "http://localhost:3103",
    trace: "on-first-retry",
  },

  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],

  webServer: {
    command: "pnpm exec next dev --port 3103",
    url: "http://localhost:3103/api/config",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    env: {
      // Unresolvable on purpose -- see the note above.
      APP_API_BASE_URL: "http://e2e.invalid",
      APP_URL: "http://localhost:3103",
      APP_SENTINEL_URL: "http://sentinel.e2e.invalid",
      APP_SENTINEL_GOOGLE_CLIENT_ID: "e2e-client-id",
      // The mint route refuses to run without this. Its value is never used:
      // `auth.ts` intercepts the route before it can reach any Sentinel.
      APP_SENTINEL_SERVICE_KEY: "e2e-not-a-real-key",
    },
  },
});
