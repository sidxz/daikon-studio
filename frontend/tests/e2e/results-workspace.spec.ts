import { expect, test } from "@playwright/test";
import { RUN_ID, installApiMock } from "./api-mock";
import { clearBrowserState, installAuth, signIn } from "./auth";

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem("theme", "light"));
  await installAuth(page);
  await installApiMock(page);
  await signIn(page);
  await page.goto(`/runs/${RUN_ID}`);
});

test.afterEach(async ({ page }) => {
  await clearBrowserState(page);
});

test("scrolling gives the table the viewport; maximizing preserves the selected and filtered rows", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  const workspace = page.getByRole("region", { name: "Prediction results table" });
  const grid = workspace.locator(".run-results-grid");
  await expect(workspace.getByRole("img", { name: "CCO", exact: true })).toBeVisible();
  const thumbnail = await workspace.getByRole("img", { name: "CCO", exact: true }).boundingBox();
  expect(thumbnail?.width).toBe(80);
  const initialHeight = (await grid.boundingBox())?.height ?? 0;
  await grid.hover();
  await page.mouse.wheel(0, 500);
  await expect.poll(async () => (await workspace.boundingBox())?.y).toBeLessThanOrEqual(9);
  await expect
    .poll(async () => (await grid.boundingBox())?.height ?? 0)
    .toBeGreaterThan(initialHeight + 100);
  const scrolledGrid = await grid.boundingBox();
  expect((scrolledGrid?.y ?? 0) + (scrolledGrid?.height ?? 0)).toBeLessThanOrEqual(900);
  await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
  await expect(workspace.getByRole("button", { name: "Maximize", exact: true })).toBeInViewport();
  await page.screenshot({ path: "test-results/results-scrolled.png" });

  const selectedRow = workspace
    .getByRole("row")
    .filter({ has: page.getByRole("img", { name: "CCO", exact: true }) });
  await selectedRow.getByRole("checkbox").check();
  await workspace.getByLabel(/within applicability domain/i).click();
  await expect(workspace.getByRole("img", { name: "CCF", exact: true })).toBeHidden();
  await workspace.getByRole("button", { name: "Maximize", exact: true }).click();
  await expect(workspace.getByRole("button", { name: "Restore", exact: true })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  const expanded = await workspace.boundingBox();
  expect(expanded?.x).toBe(0);
  expect(expanded?.y).toBe(0);
  expect(expanded?.width).toBe(1440);
  expect(expanded?.height).toBe(900);
  const dashboardLink = page.getByRole("link", { name: "Dashboard", exact: true }).first();
  expect(await dashboardLink.evaluate((link) => Boolean(link.closest("[inert]")))).toBe(true);
  await expect(workspace.getByText("1 selected", { exact: true })).toBeVisible();
  await expect(selectedRow.getByRole("checkbox")).toBeChecked();
  await expect(workspace.getByLabel(/within applicability domain/i)).toBeChecked();
  await page.screenshot({ path: "test-results/results-maximized.png" });

  // Inspecting a compound in the expanded grid must close just the inspector.
  await workspace.getByRole("button", { name: "Inspect compound at row 8", exact: true }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toBeHidden();
  await expect(workspace.getByRole("button", { name: "Restore", exact: true })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(workspace.getByRole("button", { name: "Maximize", exact: true })).toBeVisible();
  expect(await dashboardLink.evaluate((link) => Boolean(link.closest("[inert]")))).toBe(false);
  await expect(selectedRow.getByRole("checkbox")).toBeChecked();
});

test("the expanded table fits a small screen and remains usable after resizing", async ({
  page,
}) => {
  await page.setViewportSize({ width: 375, height: 812 });
  const workspace = page.getByRole("region", { name: "Prediction results table" });
  await workspace.getByRole("button", { name: "Maximize", exact: true }).click();
  await expect(workspace.getByRole("button", { name: "Export", exact: true })).toBeInViewport();
  await expect(workspace.getByRole("button", { name: "Restore", exact: true })).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(375);
  await page.screenshot({ path: "test-results/results-mobile-maximized.png" });
  await page.setViewportSize({ width: 1024, height: 768 });
  await expect.poll(async () => (await workspace.boundingBox())?.width).toBe(1024);
  await workspace.getByRole("button", { name: "Restore", exact: true }).click();
  await expect(workspace.getByRole("button", { name: "Maximize", exact: true })).toBeInViewport();
});
