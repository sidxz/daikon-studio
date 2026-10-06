import type { DatasetResponse, ScorecardResponse } from "@/shared/lib/api/model";
import { type Locator, type Page, expect, test } from "@playwright/test";
import { PROTOCOL_ID, RUN_ID, installApiMock } from "./api-mock";
import { clearBrowserState, installAuth, signIn } from "./auth";

const DATASET_ID = "33333333-3333-4333-8333-333333333333";
const dataset: DatasetResponse = {
  id: DATASET_ID,
  workspace_id: "00000000-0000-4000-8000-000000000001",
  name: "Solubility reference",
  structure_column: "smiles",
  targets: [{ column: "logS", kind: "numeric" }],
  split: { strategy: "scaffold", seed: 42 },
  content_hash: "test-content-hash",
  snapshot_uri: "memory://dataset",
  row_count: 1128,
  validation_report: {
    total_rows: 1128,
    valid_rows: 1128,
    invalid: [],
    conflicting: [],
    duplicates_collapsed: 0,
    salts_flagged: 0,
    duplicate_spread: {},
  },
  version: 1,
  created_at: "2026-10-05T12:00:00Z",
  can_delete: false,
  can_edit: false,
  id_column: null,
  created_by: null,
  folder_id: null,
};

const scorecard = (target: string): ScorecardResponse => ({
  target,
  joint_model: false,
  primary_metric: "rmse",
  primary_metric_ci: null,
  primary_metric_bootstrap: null,
  prediction_kind: "numeric",
  metrics: { rmse: 0.52 },
  validation_metrics: null,
  metrics_undefined: {},
  engine_id: "ecfp4-xgboost",
  conditions: {},
  baseline_engine_id: "ecfp4-randomforest",
  baseline_conditions: {},
  baseline_metrics: { rmse: 0.61 },
  baseline_is_self: false,
  random_split_metrics: null,
  random_split_unavailable: null,
  random_split_metrics_undefined: {},
  noise_floor: null,
  worst_rows: [],
  applicability_coverage: null,
  unit: null,
  direction: null,
  split_strategy: "random",
  cutoff: null,
  baseline_cutoff: null,
  cutoff_note: null,
  parity: [],
  parity_sampled_from: null,
  residual_histogram: null,
  error_by_similarity: [],
  scaffold_errors: [],
  calibration: [],
});

async function library(page: Page) {
  await page.addInitScript(() => localStorage.setItem("theme", "light"));
  await installAuth(page);
  await installApiMock(page);
  await page.route("**/api/config", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      json: { ...(await response.json()), chemcellarUrl: "http://chemcellar.e2e.invalid" },
    });
  });
  await page.route("**/api/v1/datasets**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    const body = path.endsWith("/profile")
      ? { status: "computing", started_at: new Date().toISOString(), compounds: 1128 }
      : path.endsWith("/compounds")
        ? { total: 0, items: [] }
        : path.endsWith(DATASET_ID)
          ? dataset
          : { items: [dataset], next_cursor: null };
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(body) });
  });
  await page.route("**/api/v1/engines**", (route) =>
    route.fulfill({ contentType: "application/json", body: "[]" }),
  );
  await page.route("**/api/v1/chemcellar/**", (route) =>
    route.fulfill({ contentType: "application/json", body: "[]" }),
  );
  await page.route("**/api/v1/protocols/*/scorecard", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify([scorecard("logS"), scorecard("pIC50")]),
    }),
  );
  await page.route("**/api/v1/collections/layout-collection", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        id: "layout-collection",
        name: "Follow-up candidates",
        member_count: 12,
        created_at: dataset.created_at,
        derived_from_run_id: RUN_ID,
        provenance: { generation_method: "ai_predicted" },
      }),
    }),
  );
  await page.route("**/api/v1/sweeps/layout-sweep", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ id: "layout-sweep", name: "Solubility sweep", runs: [] }),
    }),
  );
  await signIn(page);
}

async function checkUnderline(tab: Locator) {
  await expect
    .poll(() =>
      tab.evaluate((element) => {
        const style = getComputedStyle(element);
        return {
          radius: style.borderRadius,
          background: style.backgroundColor,
          bottom: style.borderBottomWidth,
          shadow: style.boxShadow,
        };
      }),
    )
    .toEqual({ radius: "0px", background: "rgba(0, 0, 0, 0)", bottom: "2px", shadow: "none" });
}

test.afterEach(async ({ page }) => clearBrowserState(page));

test("index, detail and creation pages share the shell margins on desktop and mobile", async ({
  page,
}) => {
  await library(page);
  const routes = [
    ["/", "Dashboard"],
    ["/datasets", "Datasets"],
    ["/protocols", "Protocols"],
    ["/runs", "Runs"],
    ["/sweeps", "Sweeps"],
    ["/collections", "Collections"],
    ["/engines", "Engines"],
    ["/runners", "Runners"],
    ["/settings", "Settings"],
    [`/datasets/${DATASET_ID}`, dataset.name],
    [`/protocols/${PROTOCOL_ID}`, "E2E Solubility Published"],
    [`/runs/${RUN_ID}`, "E2E Solubility"],
    ["/sweeps/layout-sweep", "Solubility sweep"],
    ["/collections/layout-collection", "Follow-up candidates"],
    ["/datasets/new", "New dataset"],
    ["/protocols/new", "Train a protocol"],
    ["/runs/new", "Run a protocol"],
    ["/sweeps/new", "New sweep"],
  ] as const;
  for (const width of [1440, 375]) {
    await page.setViewportSize({ width, height: 1000 });
    for (const [route, name] of routes) {
      await page.goto(route);
      const heading = page.getByRole("heading", { name, level: 1, exact: true });
      await expect(heading).toBeVisible();
      const metrics = await heading.evaluate((element) => {
        const main = element.closest("main");
        if (!main) throw new Error("Page heading is outside the app shell");
        const rect = main.getBoundingClientRect();
        const style = getComputedStyle(main);
        const root = main.firstElementChild?.getBoundingClientRect();
        return {
          left: element.getBoundingClientRect().left,
          expectedLeft: rect.left + Number.parseFloat(style.paddingLeft),
          rootWidth: root?.width,
          expectedWidth:
            rect.width -
            Number.parseFloat(style.paddingLeft) -
            Number.parseFloat(style.paddingRight),
          scrollWidth: document.documentElement.scrollWidth,
        };
      });
      expect(metrics.left, route).toBeCloseTo(metrics.expectedLeft, 0);
      expect(metrics.rootWidth, route).toBeCloseTo(metrics.expectedWidth, 0);
      expect(metrics.scrollWidth, route).toBe(width);
    }
  }
});

test("detail, target and source tabs use underlines and support keyboard selection in both themes", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await library(page);
  await page.goto(`/datasets/${DATASET_ID}`);
  const overview = page.getByRole("tab", { name: "Overview", exact: true });
  await expect(overview).toHaveAttribute("aria-selected", "true");
  await checkUnderline(overview);
  await overview.focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("tab", { name: "Diversity", exact: true })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(page.getByText(/Computing/).first()).toBeVisible();
  await page.getByRole("tab", { name: "Compounds", exact: true }).click();
  await expect(page.getByRole("tabpanel")).toBeVisible();
  await page.screenshot({ path: "test-results/layout-dataset-tabs.png", fullPage: true });

  await page.goto(`/protocols/${PROTOCOL_ID}`);
  const target = page.getByRole("tab", { name: "logS", exact: true });
  await expect(target).toBeVisible();
  await checkUnderline(target);
  await target.focus();
  await page.keyboard.press("End");
  await expect(page.getByRole("tab", { name: "pIC50", exact: true })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await page.getByRole("tab", { name: "pIC50", exact: true }).blur();
  await expect
    .poll(() => target.evaluate((element) => getComputedStyle(element).borderBottomColor))
    .toBe("rgba(0, 0, 0, 0)");
  await page.screenshot({ path: "test-results/layout-protocol-targets.png", fullPage: true });

  await page.goto("/runs/new");
  const source = page.getByRole("tab", { name: "Upload CSV", exact: true });
  await expect(source).toBeVisible();
  await checkUnderline(source);
  await source.focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("tab", { name: "From ChemCellar", exact: true })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await page.getByRole("tab", { name: "From ChemCellar", exact: true }).blur();
  await expect
    .poll(() => source.evaluate((element) => getComputedStyle(element).borderBottomColor))
    .toBe("rgba(0, 0, 0, 0)");
  await page.screenshot({ path: "test-results/layout-creation.png", fullPage: true });

  await page.goto(`/runs/${RUN_ID}`);
  const results = page.getByRole("tab", { name: "Results", exact: true });
  await expect(results).toBeVisible();
  await checkUnderline(results);
  await page.setViewportSize({ width: 375, height: 812 });
  await results.focus();
  await page.keyboard.press("End");
  const details = page.getByRole("tab", { name: "Run details", exact: true });
  await expect(details).toHaveAttribute("aria-selected", "true");
  await expect(details).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(375);
  await page.getByRole("button", { name: "Toggle theme" }).click();
  await details.blur();
  await checkUnderline(details);
  await page.screenshot({ path: "test-results/layout-run-tabs-dark-mobile.png", fullPage: true });
});
