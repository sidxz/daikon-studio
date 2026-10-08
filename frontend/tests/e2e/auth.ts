import type { Page } from "@playwright/test";

/**
 * Getting past Duar auth without a real Google login.
 *
 * ## These tokens are not credentials
 *
 * The strings minted below are JWT-*shaped* but unsigned: `header.payload.`
 * plus a literal placeholder where a signature belongs. That is enough for the
 * client, which only base64-decodes the payload and checks `exp` — it never
 * verifies a signature, because verification is the backend's job.
 *
 * Nothing here weakens that. The backend validates both tokens as RS256 against
 * a JWKS, so if these ever reached a real server every request would 401. They
 * cannot reach one anyway: `playwright.config.ts` points the app's API base URL
 * at a domain that does not resolve, and `api-mock.ts` intercepts every call.
 *
 * **If you are here because requests are 401ing against a real backend, the
 * 401 is correct.** Do not add a test-mode key override or an entrypoint that
 * patches JWKS resolution to make it pass — that turns a running server into
 * one that accepts forged identity, workspace, and role claims. Mock the
 * endpoint instead, or write the assertion as a pytest case against
 * `backend/tests/api/`, where the auth harness is process-local and cannot
 * outlive the test run.
 *
 * ## Isolation
 *
 * Playwright launches an ephemeral browser profile in a temp directory and
 * discards it when the run ends, so none of this touches the Chrome you use.
 * The suite also runs on port 3103 while your dev server runs on 3003, and
 * localStorage is partitioned by origin — port included — so even the storage
 * bucket is a different one. `clearBrowserState` below is a third belt on the
 * same trousers: redundant, cheap, and it makes the intent legible.
 */

const WORKSPACE_ID = "00000000-0000-4000-8000-000000000001";
const WORKSPACE_NAME = "E2E Workspace";
const USER_EMAIL = "e2e@example.test";
const USER_NAME = "E2E User";

function base64url(value: object): string {
  return Buffer.from(JSON.stringify(value))
    .toString("base64")
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}

/** An unsigned, payload-only JWT. See the module docstring for why that is safe here. */
function craftToken(claims: Record<string, unknown>): string {
  const header = base64url({ alg: "none", typ: "JWT" });
  const payload = base64url({
    exp: Math.floor(Date.now() / 1000) + 3600,
    iat: Math.floor(Date.now() / 1000),
    ...claims,
  });
  return `${header}.${payload}.e2e-unsigned`;
}

/**
 * Intercept the three auth hops and run the real login flow against them.
 *
 * Faking `localStorage` alone does not work: the SDK holds the IdP token in
 * memory only, so a page load with just a stored authz token lands in
 * `needs_reauth` rather than `authenticated`. Letting the real flow run — with
 * each external hop answered locally — is what actually reaches the dashboard.
 */
export async function installAuth(page: Page): Promise<void> {
  // Hop 1: the IdP. Answer with a redirect carrying an id_token in the
  // fragment. The payload MUST echo the `nonce` the SDK generated, or the SDK
  // rejects the response as a replay.
  await page.route("**accounts.google.com/o/oauth2/**", async (route) => {
    const url = new URL(route.request().url());
    const redirectUri =
      url.searchParams.get("redirect_uri") ?? "http://localhost:3103/auth/callback";
    const nonce = url.searchParams.get("nonce") ?? "";
    const state = url.searchParams.get("state") ?? "";

    const idToken = craftToken({
      sub: "e2e-idp-sub",
      email: USER_EMAIL,
      name: USER_NAME,
      nonce,
      aud: "e2e-client-id",
      iss: "https://accounts.google.com",
    });

    const fragment = new URLSearchParams({ id_token: idToken, state }).toString();
    await route.fulfill({ status: 302, headers: { location: `${redirectUri}#${fragment}` } });
  });

  // Hop 2: workspace discovery. Exactly one workspace, so the SDK auto-selects
  // and no picker appears between the callback and the dashboard.
  await page.route("**/authz/resolve", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        workspaces: [
          { id: WORKSPACE_ID, name: WORKSPACE_NAME, slug: "e2e-workspace", role: "editor" },
        ],
      }),
    });
  });

  // Hop 3: the same-origin BFF that would otherwise forward to Duar with a
  // service key. Answering it here is what keeps the real service key — and the
  // real Duar — out of the test entirely.
  await page.route("**/api/auth/mint", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        authz_token: craftToken({
          sub: "e2e-user-id",
          idp_sub: "e2e-idp-sub",
          wid: WORKSPACE_ID,
          wslug: "e2e-workspace",
          wrole: "editor",
          aud: "duar:authz",
        }),
        user: { email: USER_EMAIL, name: USER_NAME },
      }),
    });
  });
}

/** Sign in and land on the dashboard. */
export async function signIn(page: Page): Promise<void> {
  await page.goto("/login");
  await page.getByRole("button", { name: /continue with google/i }).click();
  await page.waitForURL((url) => !/\/login|\/auth\//.test(url.pathname), { timeout: 30_000 });
}

/**
 * Drop everything this suite wrote for the origin.
 *
 * Strictly redundant — the browser profile is thrown away when the run ends,
 * and it was never your profile or your port to begin with. Kept so that
 * "the tests clean up after themselves" is something you can read in the code
 * rather than something you have to take on trust.
 */
export async function clearBrowserState(page: Page): Promise<void> {
  await page.context().clearCookies();
  // A page that never navigated has no origin to clear, and evaluating against
  // about:blank throws — so only clear when there is something to clear.
  if (!page.url().startsWith("http")) return;
  await page.evaluate(() => {
    localStorage.clear();
    sessionStorage.clear();
  });
}
