import { expect, test } from "@playwright/test";
import { PROTOCOL_ID, installApiMock } from "./api-mock";
import { clearBrowserState, installAuth, signIn } from "./auth";

test.use({ timezoneId: "America/Chicago" });

test.afterEach(async ({ page }) => {
  await clearBrowserState(page);
});

test("run history keeps its filters usable across desktop, mobile and both themes", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.addInitScript(() => localStorage.setItem("theme", "light"));
  await installAuth(page);
  await installApiMock(page);
  await page.route("**/workspaces/*/members*", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify([{ user_id: "e2e-user-id", name: "Sid Rath" }]),
    }),
  );
  const now = Date.now();
  const runs = [
    { id: "batch-7", name: "Live check batch 7", status: "ready", count: 3 },
    { id: "nuisance", name: "Nuisance Test", status: "ready", count: 15, source: "ADMET Panel" },
    { id: "working", name: "Screening batch", status: "running", count: null },
    { id: "herg", name: "hERG Model 2", status: "ready", count: 60, source: "Dose response" },
    { id: "failed", name: "Permeability screen", status: "failed", count: null },
    { id: "large", name: "Nuisance library", status: "ready", count: 38706 },
    { id: "esol", name: "Solubility reference", status: "ready", count: 25 },
  ].map((row, index) => ({
    id: row.id,
    name: row.name,
    status: row.status,
    kind: "prediction",
    protocol_id: PROTOCOL_ID,
    requested_by: "e2e-user-id",
    created_at: new Date(now - (index < 5 ? index * 3600000 : 86400000)).toISOString(),
    metrics: row.count == null ? null : { scored_rows: row.count, uploaded_rows: row.count },
    phase: row.status === "running" ? "Predicting compounds" : null,
    progress: row.status === "running" ? 0.42 : 1,
    source: row.source
      ? { protocol_name: row.source, run_id: "source-run", run_date: "2026-03-20" }
      : null,
  }));
  const requests: URLSearchParams[] = [];
  await page.route("**/api/v1/runs?**", async (route) => {
    const params = new URL(route.request().url()).searchParams;
    requests.push(params);
    const statuses = params.getAll("status");
    const q = params.get("q")?.toLowerCase();
    const from = params.get("created_from");
    const before = params.get("created_before");
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        items: runs.filter(
          (run) =>
            (!statuses.length || statuses.includes(run.status)) &&
            (!q || run.name.toLowerCase().includes(q)) &&
            (!from || new Date(run.created_at) >= new Date(from)) &&
            (!before || new Date(run.created_at) < new Date(before)),
        ),
        next_cursor: null,
      }),
    });
  });
  await signIn(page);
  await page.goto("/runs?mine=0");
  await expect(page.getByRole("button", { name: "All", exact: true })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page.goto("/runs");
  await expect(page.getByRole("button", { name: "All", exact: true })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page.reload();
  await expect(page.getByRole("button", { name: "All", exact: true })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.getByRole("link", { name: /Nuisance Test/ })).toContainText(
    "ChemCellar · ADMET Panel",
  );
  await expect(page.getByRole("link", { name: /Live check batch 7/ })).toContainText(
    "E2E Solubility",
  );
  await expect(page.getByRole("button", { name: "Clear", exact: true })).toBeHidden();
  await expect(page.getByText("7 runs shown", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("link", { name: /Screening batch/ }).getByLabel("Compound count unavailable"),
  ).toBeVisible();
  await page.screenshot({ path: "test-results/runs-desktop-light.png", fullPage: true });
  await page.getByRole("button", { name: "Toggle theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.screenshot({ path: "test-results/runs-desktop-dark.png", fullPage: true });
  await page.getByRole("button", { name: "Toggle theme" }).click();

  await page.setViewportSize({ width: 375, height: 812 });
  await expect(page.getByRole("searchbox", { name: "Search runs or protocols" })).toBeInViewport();
  await expect(page.getByRole("combobox", { name: "Status" })).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(375);
  await page.screenshot({ path: "test-results/runs-mobile.png", fullPage: true });
  await page.getByRole("combobox", { name: "Status" }).click();
  await page.getByRole("option", { name: "Ready", exact: true }).click();
  await page.getByRole("searchbox", { name: "Search runs or protocols" }).fill("Nuisance");
  await expect(page).toHaveURL(/status=ready.*q=Nuisance/);
  await expect(page.getByText("2 runs shown", { exact: true })).toBeVisible();
  expect(requests.at(-1)?.getAll("status")).toEqual(["ready"]);
  expect(requests.at(-1)?.get("q")).toBe("Nuisance");
  await page.getByRole("button", { name: "Clear", exact: true }).click();
  await expect(page).toHaveURL(/\/runs$/);
  await expect(page.getByRole("searchbox", { name: "Search runs or protocols" })).toHaveValue("");
  await expect(page.getByText("7 runs shown", { exact: true })).toBeVisible();
  await expect(page.getByRole("link", { name: /Nuisance Test/ })).toHaveAttribute(
    "href",
    "/runs/nuisance",
  );

  const day = await page.evaluate((iso) => {
    const date = new Date(iso);
    return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
  }, runs[5].created_at);
  await page.getByRole("button", { name: /^Date range:/ }).click();
  await page.getByLabel("From", { exact: true }).fill(day);
  await page.getByLabel("To", { exact: true }).fill(day);
  await page.screenshot({ path: "test-results/runs-date-filter.png", fullPage: true });
  await page.getByRole("button", { name: "Apply dates" }).click();
  await expect(page.getByText("2 runs shown", { exact: true })).toBeVisible();
  await expect(page.getByRole("link", { name: /Live check batch 7/ })).toBeHidden();
  expect(requests.at(-1)?.get("mine")).toBeNull();
  expect(requests.at(-1)?.get("created_from")).toBeTruthy();
  expect(requests.at(-1)?.get("created_before")).toBeTruthy();
  await page.getByRole("button", { name: /^Date range:/ }).click();
  await page.getByLabel("To", { exact: true }).fill("2020-01-01");
  await expect(page.getByRole("button", { name: "Apply dates" })).toBeDisabled();
  await expect(page.getByRole("alert").filter({ hasText: "end on or after" })).toBeVisible();
  await page.getByRole("button", { name: "All dates" }).click();
  await expect(page.getByText("7 runs shown", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Started by me" }).click();
  await expect(page.getByRole("button", { name: "Started by me" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page.goto("/runs");
  await expect(page.getByRole("button", { name: "Started by me" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  expect(requests.at(-1)?.get("mine")).toBe("true");
});
