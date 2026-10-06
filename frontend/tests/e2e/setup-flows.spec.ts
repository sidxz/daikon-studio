import type {
  DatasetPreviewResponse,
  DatasetResponse,
  EngineManifestResponse,
} from "@/shared/lib/api/model";
import { type Page, expect, test } from "@playwright/test";
import { installApiMock } from "./api-mock";
import { clearBrowserState, installAuth, signIn } from "./auth";

const report = {
  total_rows: 12,
  valid_rows: 11,
  invalid: [{ row_number: 4, value: "invalid", reason: "Invalid SMILES" }],
  conflicting: [],
  duplicates_collapsed: 1,
  salts_flagged: 0,
  duplicate_spread: { activity: 0.1 },
};
const readiness = {
  row_count: 10,
  partition_counts: { train: 8, validation: 1, test: 1 },
  class_balance: [],
  warnings: [],
};
const dataset: DatasetResponse = {
  id: "dataset-reviewed",
  workspace_id: "workspace",
  name: "Activity reference",
  structure_column: "smiles",
  targets: [{ column: "activity", kind: "numeric", unit: "µM", direction: "low" }],
  split: { strategy: "scaffold", seed: 42, fractions: [0.8, 0.1, 0.1] },
  content_hash: "hash",
  snapshot_uri: "memory://snapshot",
  row_count: 10,
  validation_report: report,
  version: 1,
  created_at: "2026-10-05T12:00:00Z",
  can_delete: false,
  can_edit: true,
  id_column: "id",
  created_by: null,
  folder_id: null,
};
const condition = {
  key: "trees",
  label: "Trees",
  type: "integer",
  default: 500,
  minimum: 10,
  maximum: 1000,
  required: false,
  options: [],
  option_labels: [],
  tasks: [],
  help: "Number of trees.",
};
const engines: EngineManifestResponse[] = [
  {
    id: "forest",
    version: "1",
    name: "Reference forest",
    description: "Learns from molecular fingerprints.",
    tasks: ["regression", "binary_classification"],
    conditions: [condition],
    is_baseline: true,
    supports_multitask: false,
    lane: "default",
  },
  {
    id: "graph",
    version: "1",
    name: "Graph model",
    description: "Learns from molecular graphs.",
    tasks: ["regression", "binary_classification"],
    conditions: [{ ...condition, key: "epochs", label: "Epochs", default: 30, maximum: 100 }],
    is_baseline: false,
    supports_multitask: true,
    lane: "gpu",
  },
];
const preview: DatasetPreviewResponse = {
  id: "review-1",
  name: "assay",
  status: "succeeded",
  stage: "Ready to review",
  done: 0,
  total: 0,
  dataset_id: null,
  error: null,
  created_at: "2026-10-05T12:00:00Z",
  updated_at: "2026-10-05T12:00:01Z",
  preparation: {
    name: "assay",
    file_name: "assay.csv",
    structure_column: "smiles",
    targets: dataset.targets,
    split: dataset.split,
    id_column: "id",
    validation_report: report,
    readiness,
    expires_at: "2026-10-06T12:00:00Z",
  },
};

async function setup(page: Page) {
  await page.addInitScript(() => localStorage.setItem("theme", "light"));
  await installAuth(page);
  await installApiMock(page);
  await page.route("**/api/v1/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
    if (path === "/api/v1/engines") return json(engines);
    if (path === "/api/v1/datasets")
      return json({ items: [dataset], next_cursor: null, total_count: null });
    if (path === `/api/v1/datasets/${dataset.id}`) return json(dataset);
    if (path.endsWith("/readiness")) return json(readiness);
    if (path.endsWith("/profile"))
      return json({ started_at: "2026-10-05T12:00:00Z", compounds: 10 }, 202);
    if (path === "/api/v1/datasets/uploads") return json({ upload_ref: "upload-1" }, 201);
    if (path === "/api/v1/datasets/previews")
      return json({ ...preview, status: "running", preparation: null }, 202);
    if (path === "/api/v1/datasets/previews/review-1") return json(preview);
    if (path.endsWith("/freeze")) return json(dataset, 201);
    return route.fallback();
  });
  await signIn(page);
}

test.afterEach(async ({ page }) => clearBrowserState(page));

async function upload(page: Page, name = "assay.csv") {
  await page.locator('input[type="file"]').setInputFiles({
    name,
    mimeType: "text/csv",
    buffer: Buffer.from("id,smiles,activity\nA1,CCO,1.2\nA2,CCN,2.4\nA3,CCC,3.7\n"),
  });
  await expect(page.getByLabel("Dataset name")).toBeVisible();
}

test("dataset preparation is reviewed before freeze, survives reload, and exports excluded rows", async ({
  page,
}) => {
  await setup(page);
  await page.goto("/datasets/new");
  await upload(page);
  await page.getByLabel("Unit (optional)").fill("µM");
  await page.getByRole("combobox", { name: "Preferred direction" }).click();
  await page.getByRole("option", { name: "Lower is better" }).click();
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(
    page.getByText("Intended split: 80% training · 10% validation · 10% test"),
  ).toBeVisible();
  let freezeCalls = 0;
  page.on("request", (request) => {
    if (request.url().endsWith("/freeze")) freezeCalls++;
  });
  await page.getByRole("button", { name: "Prepare review", exact: true }).click();
  await expect(page.getByText("Review your dataset", { exact: true })).toBeVisible();
  expect(freezeCalls).toBe(0);
  await expect(page.getByRole("button", { name: "Create dataset", exact: true })).toBeEnabled();
  await expect(page.getByText("Invalid SMILES", { exact: true })).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download issue report" }).click();
  expect((await download).suggestedFilename()).toBe("dataset-issues.csv");
  await page.reload();
  await expect(page.getByLabel("Dataset name")).toHaveValue("assay");
  await expect(page.getByText("Review your dataset", { exact: true })).toBeVisible();
  await expect(page.getByText("assay.csv", { exact: true })).toBeVisible();
  await page.screenshot({ path: "test-results/dataset-review-desktop.png", fullPage: true });
  await page.setViewportSize({ width: 375, height: 812 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  await page.screenshot({ path: "test-results/dataset-review-mobile.png", fullPage: true });
  await page.getByRole("button", { name: "Create dataset", exact: true }).click();
  await expect(page).toHaveURL(/\/datasets\/dataset-reviewed$/);
  expect(freezeCalls).toBe(1);
});

test("replacing a CSV keeps compatible target metadata and column roles", async ({ page }) => {
  await setup(page);
  await page.goto("/datasets/new");
  await upload(page);
  await page.getByLabel("Unit (optional)").fill("µM");
  await page.getByRole("combobox", { name: "Preferred direction" }).click();
  await page.getByRole("option", { name: "Lower is better" }).click();
  await upload(page, "corrected.csv");
  await expect(page.getByLabel("Unit (optional)")).toHaveValue("µM");
  await expect(page.getByRole("combobox", { name: "Preferred direction" })).toContainText(
    "Lower is better",
  );
  await expect(page.getByRole("combobox", { name: "Role for id", exact: true })).toContainText(
    "Identifier",
  );
  await expect(page.getByLabel("Dataset name")).toHaveValue("assay");
});

test("training summaries show all planned work and reveal an invalid baseline setting", async ({
  page,
}) => {
  await setup(page);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(`/protocols/new?dataset=${dataset.id}`);
  await page.getByRole("radio", { name: "Graph model", exact: true }).click();
  const plan = page.getByRole("complementary", { name: "Training plan" });
  await expect(plan.getByText("3 training stages", { exact: true })).toBeVisible();
  await expect(
    plan.getByText("Train Graph model again on a random split for comparison.", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Change baseline", exact: true }).click();
  await page
    .getByRole("button", { name: /Advanced settings/ })
    .last()
    .click();
  await page.getByLabel("Trees").fill("2000");
  await page.getByRole("button", { name: "Hide baseline settings", exact: true }).click();
  await expect(page.getByRole("button", { name: "Train", exact: true })).toBeDisabled();
  await page
    .getByRole("button", { name: "Baseline: Trees: Must be between 10 and 1000.", exact: true })
    .click();
  await expect(page.getByLabel("Trees")).toBeVisible();
  await expect(page.getByLabel("Trees")).toBeFocused();
  await page.getByLabel("Trees").fill("500");
  await expect(page.getByRole("button", { name: "Train", exact: true })).toBeEnabled();
  expect(await plan.evaluate((element) => element.scrollHeight <= element.clientHeight + 1)).toBe(
    true,
  );
  await page.screenshot({ path: "test-results/training-setup-desktop.png", fullPage: true });
  await page.setViewportSize({ width: 1440, height: 700 });
  await page.evaluate(() => {
    document.documentElement.style.fontSize = "125%";
  });
  expect(await plan.evaluate((element) => element.scrollHeight <= element.clientHeight + 1)).toBe(
    true,
  );
  const train = page.getByRole("button", { name: "Train", exact: true });
  await train.scrollIntoViewIfNeeded();
  await expect(train).toBeInViewport();
  await page.screenshot({ path: "test-results/training-setup-desktop-short.png", fullPage: true });
  await page.evaluate(() => {
    document.documentElement.style.fontSize = "";
  });
  await page.setViewportSize({ width: 375, height: 812 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  await expect(page.getByRole("button", { name: "Train", exact: true })).toBeVisible();
  await page.screenshot({ path: "test-results/training-setup-mobile.png", fullPage: true });
});

test("dataset search reaches older matches through the server", async ({ page }) => {
  await setup(page);
  const queries: string[] = [];
  await page.route("**/api/v1/datasets?**", async (route) => {
    const q = new URL(route.request().url()).searchParams.get("q");
    if (!q) return route.fallback();
    queries.push(q);
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        items: [{ ...dataset, id: "older", name: "Older assay" }],
        next_cursor: null,
      }),
    });
  });
  await page.goto("/protocols/new");
  await page.getByRole("button", { name: "Dataset", exact: true }).click();
  await page.getByRole("combobox", { name: "Search datasets" }).fill("Older");
  await expect(page.getByRole("option", { name: /Older assay/ })).toBeVisible();
  expect(queries).toContain("Older");
});

test("Train submits the model and baseline settings shown in the plan", async ({ page }) => {
  await setup(page);
  let submitted: Record<string, unknown> | undefined;
  await page.route("**/api/v1/protocols", async (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    submitted = route.request().postDataJSON();
    return route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify({ id: "training-new", status: "queued" }),
    });
  });
  await page.route("**/api/v1/runs/training-new", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        id: "training-new",
        status: "running",
        progress: 0,
        phase: "Starting",
      }),
    }),
  );
  await page.goto(`/protocols/new?dataset=${dataset.id}`);
  await page.getByRole("radio", { name: "Graph model", exact: true }).click();
  await page.getByRole("button", { name: /Advanced settings/ }).click();
  await page.getByLabel("Epochs").fill("50");
  await page.getByRole("button", { name: "Change baseline", exact: true }).click();
  await page
    .getByRole("button", { name: /Advanced settings/ })
    .last()
    .click();
  await page.getByLabel("Trees").fill("400");
  const plan = page.getByRole("complementary", { name: "Training plan" });
  await expect(plan.getByText("50", { exact: true })).toBeVisible();
  await expect(plan.getByText("400", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Train", exact: true }).click();
  await expect
    .poll(() => submitted)
    .toEqual({
      name: "Activity reference · Graph model",
      dataset_id: dataset.id,
      engine_id: "graph",
      conditions: { epochs: 50 },
      baseline_engine_id: "forest",
      baseline_conditions: { trees: 400 },
      tune_cutoffs: false,
    });
});
