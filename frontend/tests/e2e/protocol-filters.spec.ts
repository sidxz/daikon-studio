import { type Page, expect, test } from "@playwright/test";
import { installApiMock } from "./api-mock";
import { clearBrowserState, installAuth, signIn } from "./auth";

async function library(page: Page) {
  await page.addInitScript(() => localStorage.setItem("theme", "light"));
  await installAuth(page);
  await installApiMock(page);
  const protocols = [
    {
      name: "Solubility reference",
      target: "logS",
      engine: "ecfp4-xgboost",
      status: "draft",
      folder: "research",
      owner: "e2e-user-id",
    },
    {
      name: "hERG panel",
      target: "herg_blocker",
      engine: "chemprop-dmpnn",
      status: "published",
      folder: "research",
      owner: "colleague",
    },
    {
      name: "Solubility comparison",
      target: "logS",
      engine: "ecfp4-xgboost",
      status: "draft",
      folder: "research",
      owner: "e2e-user-id",
    },
    {
      name: "Permeability screen",
      target: "Papp",
      engine: "chemprop-dmpnn",
      status: "published",
      folder: null,
      owner: "e2e-user-id",
    },
  ].map((spec, index) => ({
    id: `protocol-${index}`,
    name: spec.name,
    readouts: [
      {
        name: spec.target,
        type: "numeric",
        unit: null,
        direction: null,
        description: `Predicted ${spec.target}`,
      },
    ],
    engine_id: spec.engine,
    status: spec.status,
    protocol_version: 1,
    created_at: "2026-10-05T12:00:00Z",
    created_by: spec.owner,
    folder_id: spec.folder,
  }));
  await page.route("**/api/v1/engines**", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify([
        { id: "ecfp4-xgboost", name: "XGBoost" },
        { id: "chemprop-dmpnn", name: "Chemprop D-MPNN" },
      ]),
    }),
  );
  await page.route("**/api/v1/runs**", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ items: [], next_cursor: null }),
    }),
  );
  await page.route("**/api/v1/folders**", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        can_edit: true,
        items: [
          "Research",
          "Archive",
          "Follow-up",
          "Validation",
          "Kinase selectivity",
          "Metabolic stability",
        ].map((name, index) => ({
          id: index === 0 ? "research" : `folder-${index}`,
          name,
          kind: "protocol",
          created_by: "e2e-user-id",
          item_count: index === 0 ? 3 : 0,
        })),
      }),
    }),
  );
  await page.route("**/api/v1/protocols**", async (route) => {
    const params = new URL(route.request().url()).searchParams;
    const q = params.get("q")?.toLowerCase();
    const matches = protocols
      .filter(
        (protocol) =>
          (!params.has("folder_id") || protocol.folder_id === params.get("folder_id")) &&
          (!params.has("mine") || protocol.created_by === "e2e-user-id") &&
          (!params.has("engine_id") || protocol.engine_id === params.get("engine_id")) &&
          (!params.has("status") || protocol.status === params.get("status")) &&
          (!q ||
            protocol.name.toLowerCase().includes(q) ||
            protocol.readouts.some((readout) => readout.name.toLowerCase().includes(q))),
      )
      .reverse();
    const offset = Number(params.get("cursor") ?? 0);
    const limit = q ? 1 : 50;
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        items: matches.slice(offset, offset + limit),
        next_cursor: offset + limit < matches.length ? String(offset + limit) : null,
      }),
    });
  });
  await signIn(page);
  await page.goto("/protocols");
  await expect(page.getByRole("link", { name: "Solubility reference", exact: true })).toBeVisible();
}

test.afterEach(async ({ page }) => clearBrowserState(page));

test("protocol filters combine with folders, paginate and survive reload on desktop and mobile", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await library(page);
  const folders = page.getByRole("navigation", { name: "Folders", exact: true });
  const search = page.getByRole("searchbox", { name: "Search protocols or targets" });
  await folders.getByRole("button", { name: /^Research/ }).click();
  await search.fill("logS");
  await expect(page).toHaveURL((url) => url.searchParams.get("q") === "logS");
  await expect(
    page.getByRole("link", { name: "Solubility comparison", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: "Solubility reference", exact: true })).toBeHidden();
  await page.getByRole("button", { name: "Load more", exact: true }).click();
  await expect(page.getByRole("link", { name: "Solubility reference", exact: true })).toBeVisible();
  await page.getByRole("combobox", { name: "Engine", exact: true }).click();
  await page.getByRole("option", { name: "XGBoost", exact: true }).click();
  await page.getByRole("combobox", { name: "Protocol status" }).click();
  await page.getByRole("option", { name: "Draft", exact: true }).click();
  await page
    .getByRole("group", { name: "Protocol owner" })
    .getByRole("button", { name: "Created by me", exact: true })
    .click();
  await expect(page).toHaveURL(
    (url) =>
      url.searchParams.get("folder") === "research" &&
      url.searchParams.get("q") === "logS" &&
      url.searchParams.get("engine_id") === "ecfp4-xgboost" &&
      url.searchParams.get("status") === "draft" &&
      url.searchParams.get("mine") === "1",
  );
  await page.reload();
  await expect(search).toHaveValue("logS");
  await expect(page.getByRole("combobox", { name: "Engine", exact: true })).toContainText(
    "XGBoost",
  );
  await expect(page.getByRole("combobox", { name: "Protocol status" })).toContainText("Draft");
  await expect(page.getByRole("button", { name: "Created by me", exact: true })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page.screenshot({ path: "test-results/protocol-search-light.png", fullPage: true });
  await page.getByRole("combobox", { name: "Protocol status" }).click();
  await page.getByRole("option", { name: "Published", exact: true }).click();
  await expect(page.getByText("No protocols match these filters.", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Clear filters", exact: true }).click();
  await expect(page).toHaveURL(/\/protocols\?folder=research$/);
  await expect(search).toHaveValue("");
  await expect(folders.getByRole("button", { name: /^Research/ })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await search.fill("HERG_BLOCKER");
  await page.getByRole("combobox", { name: "Engine", exact: true }).click();
  await page.getByRole("option", { name: "Chemprop D-MPNN", exact: true }).click();
  await expect(page).toHaveURL(
    (url) =>
      url.searchParams.get("q") === "HERG_BLOCKER" &&
      url.searchParams.get("engine_id") === "chemprop-dmpnn",
  );
  await expect(page.getByRole("link", { name: "hERG panel", exact: true })).toBeVisible();
  await page.setViewportSize({ width: 375, height: 812 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(375);
  await expect(folders.getByRole("button", { name: /^Research/ })).toBeVisible();
  await page.screenshot({ path: "test-results/protocol-search-mobile.png", fullPage: true });
  await page.getByRole("button", { name: "Toggle theme" }).click();
  await page.screenshot({ path: "test-results/protocol-search-dark.png", fullPage: true });
  await page.getByRole("button", { name: "Clear", exact: true }).click();
  await expect(page).toHaveURL(/\/protocols\?folder=research$/);
  await expect(search).toHaveValue("");
  await folders.getByRole("button", { name: "All protocols", exact: true }).click();
  await expect(page.getByRole("link", { name: "Permeability screen", exact: true })).toBeVisible();
});
