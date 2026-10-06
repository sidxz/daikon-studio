import { type Page, expect, test } from "@playwright/test";
import { installApiMock } from "./api-mock";
import { clearBrowserState, installAuth, signIn } from "./auth";

async function library(page: Page, canEdit = true) {
  await page.addInitScript(() => localStorage.setItem("theme", "light"));
  await installAuth(page);
  await installApiMock(page);
  let folders = [
    "Nuisance screening",
    "Solubility",
    "Ion channels",
    "Kinase selectivity",
    "Permeability",
    "Metabolic stability",
    "Follow-up candidates",
    "Archive",
  ].map((name, index) => ({
    id: `folder-${index}`,
    name,
    kind: "dataset",
    created_by: "e2e-user-id",
  }));
  const datasets = [
    "Solubility reference",
    "hERG training",
    "Permeability screen",
    "Solubility comparison",
  ].map((name, index) => ({
    id: `dataset-${index}`,
    name,
    targets: [
      {
        column: index === 1 ? "herg_blocker" : index === 2 ? "Papp" : "logS",
        kind: index === 1 ? "binary" : "numeric",
      },
    ],
    row_count: [9955, 13420, 1092, 2000][index],
    split: { strategy: index === 1 ? "random" : "scaffold", seed: 42 },
    created_at: "2026-10-05T12:00:00Z",
    created_by: null,
    folder_id: index === 0 || index === 3 ? "folder-0" : (null as string | null),
  }));
  const moves: { item: string; folder: string }[] = [];
  await page.route("**/api/v1/folders**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const id = url.pathname.split("/").at(-1);
    let result: unknown;
    if (request.method() === "POST") {
      const folder = {
        id: "new-folder",
        name: request.postDataJSON().name,
        kind: "dataset",
        created_by: "e2e-user-id",
      };
      folders.push(folder);
      result = { ...folder, item_count: 0 };
    } else if (request.method() === "PATCH") {
      const folder = folders.find((folder) => folder.id === id);
      if (!folder) throw new Error("Missing mock folder");
      folder.name = request.postDataJSON().name;
      result = { ...folder, item_count: 0 };
    } else if (request.method() === "DELETE") {
      folders = folders.filter((folder) => folder.id !== id);
      for (const dataset of datasets) if (dataset.folder_id === id) dataset.folder_id = null;
      await route.fulfill({ status: 204 });
      return;
    } else {
      result = {
        can_edit: canEdit,
        items:
          url.searchParams.get("kind") === "protocol"
            ? []
            : folders.map((folder) => ({
                ...folder,
                item_count: datasets.filter((dataset) => dataset.folder_id === folder.id).length,
              })),
      };
    }
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(result) });
  });
  await page.route("**/api/v1/datasets**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === "PUT") {
      const id = url.pathname.split("/").at(-2);
      const dataset = datasets.find((dataset) => dataset.id === id);
      if (!dataset) throw new Error("Missing mock dataset");
      dataset.folder_id = request.postDataJSON().folder_id;
      moves.push({ item: dataset.id, folder: dataset.folder_id as string });
      await route.fulfill({ contentType: "application/json", body: JSON.stringify(dataset) });
      return;
    }
    const folder = url.searchParams.get("folder_id");
    const q = url.searchParams.get("q")?.toLowerCase();
    const kind = url.searchParams.get("target_kind");
    const split = url.searchParams.get("split_strategy");
    const matches = datasets
      .filter(
        (dataset) =>
          (!folder || dataset.folder_id === folder) &&
          (!q ||
            dataset.name.toLowerCase().includes(q) ||
            dataset.targets.some((target) => target.column.toLowerCase().includes(q))) &&
          (!kind || dataset.targets.some((target) => target.kind === kind)) &&
          (!split || dataset.split.strategy === split),
      )
      .reverse();
    const offset = Number(url.searchParams.get("cursor") ?? 0);
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
  await page.goto("/datasets");
  await expect(page.getByRole("link", { name: "Solubility reference", exact: true })).toBeVisible();
  return moves;
}

test.afterEach(async ({ page }) => clearBrowserState(page));

test("folder tabs preserve filtering, filing and folder management across screen sizes", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  const moves = await library(page);
  const navigation = page.getByRole("navigation", { name: "Folders", exact: true });
  await expect(navigation.getByRole("button", { name: /^Nuisance screening/ })).toBeVisible();
  await page.screenshot({ path: "test-results/dataset-tabs-light.png", fullPage: true });
  await navigation.getByRole("button", { name: /^Nuisance screening/ }).click();
  await expect(page).toHaveURL(/folder=folder-0/);
  await expect(page.getByRole("link", { name: "hERG training", exact: true })).toBeHidden();
  await page.reload();
  await expect(navigation.getByRole("button", { name: /^Nuisance screening/ })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await navigation.getByRole("button", { name: "All datasets", exact: true }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("link", { name: "hERG training", exact: true })).toBeVisible();

  const card = page
    .locator('[draggable="true"]')
    .filter({ has: page.getByRole("link", { name: "Solubility reference", exact: true }) });
  await card.dragTo(navigation.getByTestId("folder-folder-2"));
  await expect.poll(() => moves).toEqual([{ item: "dataset-0", folder: "folder-2" }]);
  await expect(navigation.getByRole("button", { name: /^Ion channels/ })).toContainText("1");

  await navigation.getByTestId("folder-folder-0").hover();
  await navigation.getByRole("button", { name: "Folder actions for Nuisance screening" }).click();
  await page.getByRole("menuitem", { name: "Rename", exact: true }).click();
  await page.getByLabel("Folder name", { exact: true }).fill("Nuisance compounds");
  await page.getByRole("dialog").getByRole("button", { name: "Save", exact: true }).click();
  await expect(navigation.getByRole("button", { name: /^Nuisance compounds/ })).toBeVisible();

  await navigation.getByRole("button", { name: "New folder", exact: true }).click();
  await page.getByLabel("Folder name", { exact: true }).fill("Validation batch");
  await page.getByRole("dialog").getByRole("button", { name: "Save", exact: true }).click();
  await navigation.getByRole("button", { name: "More folders", exact: true }).click();
  await page.getByRole("textbox", { name: "Search folders" }).fill("Validation");
  await page.getByRole("button", { name: /^Validation batch/ }).click();
  await expect(page).toHaveURL(/folder=new-folder/);
  await expect(navigation.getByRole("button", { name: /^Validation batch/ })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.getByRole("textbox", { name: "Search folders" })).toBeHidden();

  await page.setViewportSize({ width: 375, height: 812 });
  await expect(navigation.getByRole("button", { name: /^Validation batch/ })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(375);
  await page.screenshot({ path: "test-results/dataset-tabs-mobile.png", fullPage: true });
  await page.getByRole("button", { name: "Toggle theme" }).click();
  await page.screenshot({ path: "test-results/dataset-tabs-dark.png", fullPage: true });
  await navigation.getByTestId("folder-new-folder").hover();
  await navigation.getByRole("button", { name: "Folder actions for Validation batch" }).click();
  await page.getByRole("menuitem", { name: "Delete", exact: true }).click();
  await page.getByRole("alertdialog").getByRole("button", { name: "Delete", exact: true }).click();
  await expect(page).toHaveURL(/\/datasets$/);
  await expect(page.getByRole("link", { name: "Solubility reference", exact: true })).toBeVisible();
});

test("viewers can search overflow folders without seeing management actions", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await library(page, false);
  const navigation = page.getByRole("navigation", { name: "Folders", exact: true });
  await expect(navigation.getByRole("button", { name: "New folder", exact: true })).toBeHidden();
  await expect(navigation.getByRole("button", { name: /Folder actions/ })).toHaveCount(0);
  await navigation.getByRole("button", { name: "More folders", exact: true }).click();
  await page.getByRole("textbox", { name: "Search folders" }).fill("Metabolic");
  await page.getByRole("button", { name: /^Metabolic stability/ }).focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/folder=folder-5/);
  await expect(navigation.getByRole("button", { name: /^Metabolic stability/ })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(375);
});

test("dataset search combines with folders and filters, pages through matches, and survives reload", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await library(page);
  const navigation = page.getByRole("navigation", { name: "Folders", exact: true });
  const search = page.getByRole("searchbox", { name: "Search datasets or targets" });
  await navigation.getByRole("button", { name: /^Nuisance screening/ }).click();
  await search.fill("logS");
  await expect(page).toHaveURL(/folder=folder-0.*q=logS/);
  await expect(
    page.getByRole("link", { name: "Solubility comparison", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: "Solubility reference", exact: true })).toBeHidden();
  await page.getByRole("button", { name: "Load more", exact: true }).click();
  await expect(page.getByRole("link", { name: "Solubility reference", exact: true })).toBeVisible();
  await page.getByRole("combobox", { name: "Target type" }).click();
  await page.getByRole("option", { name: "Numeric targets", exact: true }).click();
  await page.getByRole("combobox", { name: "Split strategy" }).click();
  await page.getByRole("option", { name: "Scaffold split", exact: true }).click();
  await page.screenshot({ path: "test-results/dataset-search-light.png", fullPage: true });
  await page.getByRole("combobox", { name: "Target type" }).click();
  await page.getByRole("option", { name: "Binary targets", exact: true }).click();
  await expect(page.getByText("No datasets match these filters.", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Clear filters", exact: true }).click();
  await expect(page).toHaveURL(/\/datasets\?folder=folder-0$/);
  await expect(search).toHaveValue("");
  await expect(navigation.getByRole("button", { name: /^Nuisance screening/ })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.getByRole("link", { name: "Solubility reference", exact: true })).toBeVisible();

  await navigation.getByRole("button", { name: "All datasets", exact: true }).click();
  await search.fill("hERG");
  await expect(page.getByRole("link", { name: "hERG training", exact: true })).toBeVisible();
  await page.getByRole("combobox", { name: "Target type" }).click();
  await page.getByRole("option", { name: "Binary targets", exact: true }).click();
  await page.getByRole("combobox", { name: "Split strategy" }).click();
  await page.getByRole("option", { name: "Random split", exact: true }).click();
  await expect(page).toHaveURL(
    (url) =>
      url.searchParams.get("q") === "hERG" &&
      url.searchParams.get("target_kind") === "binary" &&
      url.searchParams.get("split_strategy") === "random",
  );
  await page.reload();
  await expect(search).toHaveValue("hERG");
  await expect(page.getByRole("combobox", { name: "Target type" })).toContainText("Binary targets");
  await expect(page.getByRole("combobox", { name: "Split strategy" })).toContainText(
    "Random split",
  );
  await expect(page.getByRole("link", { name: "hERG training", exact: true })).toBeVisible();
  await expect(page.getByRole("link", { name: "Solubility comparison", exact: true })).toBeHidden();
  await page.setViewportSize({ width: 375, height: 812 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(375);
  await page.screenshot({ path: "test-results/dataset-search-mobile.png", fullPage: true });
  await page.getByRole("button", { name: "Toggle theme" }).click();
  await page.screenshot({ path: "test-results/dataset-search-dark.png", fullPage: true });
  await page.getByRole("button", { name: "Clear", exact: true }).click();
  await expect(page).toHaveURL(/\/datasets$/);
  await expect(search).toHaveValue("");
});
