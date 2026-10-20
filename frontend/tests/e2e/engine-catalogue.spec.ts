import type { EngineManifestResponse } from "@/shared/lib/api/model";
import { type Page, expect, test } from "@playwright/test";
import { installApiMock } from "./api-mock";
import { clearBrowserState, installAuth, signIn } from "./auth";

const forest: EngineManifestResponse = {
  id: "ecfp4-randomforest",
  version: "1.0.0",
  name: "ECFP4 + Random Forest",
  description:
    "Morgan fingerprints with a random forest. The default comparison baseline for small-molecule models.",
  tasks: ["regression", "binary_classification"],
  structure_kinds: ["molecule"],
  lane: "default",
  is_baseline: true,
  supports_multitask: false,
  conditions: [
    {
      key: "n_estimators",
      label: "Number of trees",
      type: "integer",
      default: 500,
      required: false,
      minimum: 10,
      maximum: 2000,
      options: [],
      option_labels: [],
      help: "More trees give more stable predictions but take longer to train.",
      tasks: [],
    },
    {
      key: "positive_weighting",
      label: "Positive-class weighting",
      type: "enum",
      default: "none",
      required: false,
      minimum: null,
      maximum: null,
      options: ["none", "balanced"],
      option_labels: ["None", "Balanced"],
      help: "How much extra attention to pay to rare positives.",
      tasks: ["binary_classification"],
    },
  ],
};

const engines: EngineManifestResponse[] = [
  forest,
  {
    ...forest,
    id: "ecfp4-xgboost",
    name: "ECFP4 + XGBoost",
    description:
      "Morgan fingerprints with gradient-boosted trees. Often more accurate than random forests on structured molecular data.",
    is_baseline: false,
  },
  {
    ...forest,
    id: "chemprop-dmpnn",
    name: "Chemprop D-MPNN",
    description:
      "A graph neural network that learns molecular features directly from atoms and bonds. Trains a joint model across targets.",
    lane: "gpu",
    is_baseline: false,
    supports_multitask: true,
  },
  {
    ...forest,
    id: "protein-descriptors-randomforest",
    name: "Protein descriptors + Random Forest",
    description:
      "A random forest using protein descriptors, the comparison baseline for sequences.",
    structure_kinds: ["sequence"],
  },
  {
    ...forest,
    id: "esm2-gp",
    name: "ESM-2 embeddings + Gaussian process",
    description: "A Gaussian process trained on pretrained protein embeddings.",
    structure_kinds: ["sequence"],
    tasks: ["regression"],
    lane: "gpu",
    is_baseline: false,
    conditions: [],
  },
];

async function catalogue(page: Page) {
  await page.addInitScript(() => localStorage.setItem("theme", "light"));
  await installAuth(page);
  await installApiMock(page);
  await page.route("**/api/v1/engines", (route) => route.fulfill({ json: engines }));
  await signIn(page);
  await page.goto("/engines");
  await expect(page.getByRole("heading", { name: "Engines", exact: true })).toBeVisible();
}

test.afterEach(async ({ page }) => clearBrowserState(page));

test("categories and filters combine while preserving the selected category on reset", async ({
  page,
}) => {
  await catalogue(page);
  await expect(page.getByRole("tab", { name: "All engines 5" })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Small molecules 3" })).toBeVisible();
  const proteins = page.getByRole("tab", { name: "Proteins 2" });
  await proteins.click();
  await expect(page.getByRole("heading", { name: forest.name, exact: true })).toBeHidden();
  await expect(page.getByText("Protein baseline", { exact: true })).toBeVisible();
  await page.getByRole("combobox", { name: "Filter by compute" }).click();
  await page.getByRole("option", { name: "GPU lane", exact: true }).click();
  await expect(page.getByRole("status")).toHaveText("1 engine matching your filters");
  await page.getByRole("searchbox", { name: "Search engines" }).fill("ESM-2");
  await page.getByRole("combobox", { name: "Filter by task" }).click();
  await page.getByRole("option", { name: "Binary classification", exact: true }).click();
  await expect(page.getByRole("heading", { name: "No engines match your filters" })).toBeVisible();
  await page.getByRole("button", { name: "Clear filters", exact: true }).last().click();
  await expect(proteins).toHaveAttribute("aria-selected", "true");
  await expect(page.getByRole("status")).toHaveText("2 engines available");
  await expect(page.getByRole("searchbox", { name: "Search engines" })).toHaveValue("");
  await proteins.focus();
  await page.keyboard.press("Home");
  await expect(page.getByRole("tab", { name: "All engines 5" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(page.getByRole("status")).toHaveText("5 engines available");
});

test("card animations replay and details expose settings without overflowing desktop or mobile", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await catalogue(page);
  const card = page
    .locator('[data-slot="card"]')
    .filter({ has: page.getByRole("heading", { name: forest.name, exact: true }) });
  const figure = card.getByRole("img", { name: /four decision trees/i });
  await expect(figure).toBeVisible();
  await expect(card.getByText("Number of trees", { exact: true })).toBeHidden();
  await card.getByRole("button", { name: "Replay", exact: true }).click();
  const firstFrame = await figure.innerHTML();
  await expect.poll(() => figure.innerHTML()).not.toBe(firstFrame);
  await card.getByRole("button", { name: "How it learns" }).click();
  await expect(figure).toBeHidden();
  await card.getByRole("button", { name: "How it learns" }).click();
  await expect(figure).toBeVisible();
  await page.screenshot({ path: "test-results/engines-desktop.png", fullPage: true });

  const trigger = card.getByRole("button", { name: `View details for ${forest.name}` });
  await trigger.click();
  const dialog = page.getByRole("dialog", { name: forest.name });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByText(forest.id, { exact: true })).toBeVisible();
  await dialog.getByRole("tab", { name: "Settings (2)" }).click();
  const table = dialog.getByRole("table", { name: "Engine settings" });
  await expect(table.getByRole("row", { name: /Number of trees/ })).toContainText("500");
  await expect(table.getByRole("row", { name: /Number of trees/ })).toContainText("10–2000");
  await expect(table.getByRole("row", { name: /Positive-class weighting/ })).toContainText(
    "None · Balanced",
  );
  await expect(table).toContainText("For Binary classification");
  await page.screenshot({ path: "test-results/engines-settings.png", fullPage: true });
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(trigger).toBeFocused();
  await expect(figure).toBeVisible();

  await page.setViewportSize({ width: 375, height: 812 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(375);
  await page.screenshot({ path: "test-results/engines-mobile.png", fullPage: true });
  await trigger.click();
  await dialog.getByRole("tab", { name: "Settings (2)" }).click();
  expect(await dialog.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
  await page.screenshot({ path: "test-results/engines-settings-mobile.png", fullPage: true });
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "Toggle theme" }).click();
  await page.screenshot({ path: "test-results/engines-dark-mobile.png", fullPage: true });
});
