import type { ScorecardResponse } from "@/shared/lib/api/model";
import { type Page, expect, test } from "@playwright/test";
import { PROTOCOL_ID, installApiMock } from "./api-mock";
import { clearBrowserState, installAuth, signIn } from "./auth";

const regression: ScorecardResponse = {
  target: "logS",
  joint_model: false,
  primary_metric: "rmse",
  primary_metric_ci: [0.72, 0.88],
  primary_metric_bootstrap: null,
  prediction_kind: "value",
  metrics: { mae: 0.75, rmse: 0.791, r2: 0.78 },
  validation_metrics: { mae: 0.7, rmse: 0.75, r2: 0.81 },
  metrics_undefined: null,
  engine_id: "ecfp4-xgboost",
  conditions: { n_estimators: 100, max_depth: 6 },
  baseline_engine_id: "ecfp4-randomforest",
  baseline_conditions: { n_estimators: 100 },
  baseline_metrics: { mae: 1, rmse: 1.05, r2: 0.65 },
  baseline_is_self: false,
  random_split_metrics: { mae: 0.5, rmse: 0.6, r2: 0.9 },
  random_split_unavailable: null,
  random_split_metrics_undefined: null,
  noise_floor: 0.12,
  worst_rows: [],
  applicability_coverage: 0.82,
  unit: "log mol/L",
  direction: "high",
  split_strategy: "scaffold",
  cutoff: null,
  baseline_cutoff: null,
  cutoff_note: null,
  parity: Array.from({ length: 200 }, (_, i) => ({
    actual: i,
    predicted: i + (i % 2 ? 1 : -0.5),
    similarity: 0.2 + (i % 8) / 10,
  })),
  parity_sampled_from: null,
  residual_histogram: { edges: [-0.5, 0, 0.5, 1], counts: [100, 0, 100] },
  error_by_similarity: [],
  scaffold_errors: [],
  calibration: [],
  test_count: 200,
  regression_summary: { mean_signed_error: 0.25, absolute_error_p90: 1 },
  classification_summary: null,
  ranked_high: [],
  ranked_low: [],
  classification_by_similarity: [],
};

const binary: ScorecardResponse = {
  ...regression,
  target: "active",
  prediction_kind: "probability",
  unit: null,
  primary_metric: "mcc",
  metrics: { mcc: 0.7, balanced_accuracy: 0.85, auroc: 0.91, auprc: 0.8 },
  validation_metrics: null,
  baseline_metrics: { mcc: 0.5, auprc: 0.84 },
  primary_metric_ci: [0.6, 0.8],
  random_split_metrics: null,
  noise_floor: null,
  cutoff: 0.3,
  baseline_cutoff: 0.5,
  residual_histogram: null,
  regression_summary: null,
  parity: Array.from({ length: 200 }, (_, i) => ({
    actual: i < 50 ? 1 : 0,
    predicted: i < 40 || (i >= 50 && i < 63) ? 0.8 : 0.1,
    similarity: i < 20 ? 0.1 : i < 100 ? 0.5 : 0.9,
  })),
  calibration: [
    { lower: 0, upper: 0.3, count: 147, value: 10 / 147 },
    { lower: 0.6, upper: 1, count: 53, value: 40 / 53 },
  ],
  classification_summary: {
    true_positive: 40,
    false_negative: 10,
    false_positive: 13,
    true_negative: 137,
    precision: 40 / 53,
    recall: 0.8,
    cutoff_inclusive: true,
  },
};

// Fixtures carry server-computed diagnostics. Distinct compounds and target
// directions exercise the UI independently of the user's displayed example.
for (const card of [regression, binary]) {
  const rows = card.parity.map((point, index) => ({
    ...point,
    compound_id: `ASSAY-${index + 1}`,
    structure: `${"C".repeat((index % 10) + 1)}O${"C".repeat(Math.floor(index / 10))}`,
    test_index: index,
  }));
  card.ranked_high = [...rows].sort((a, b) => b.predicted - a.predicted).slice(0, 20);
  card.ranked_low = [...rows].sort((a, b) => a.predicted - b.predicted).slice(0, 20);
  card.worst_rows = [...rows]
    .sort((a, b) => Math.abs(b.actual - b.predicted) - Math.abs(a.actual - a.predicted))
    .slice(0, 20)
    .map((row) => ({ ...row, residual: Math.abs(row.actual - row.predicted), scaffold: "" }));
  const similarities = [...new Set(rows.map((row) => row.similarity as number))].sort();
  card.error_by_similarity = similarities.map((similarity) => {
    const group = rows.filter((row) => row.similarity === similarity);
    return {
      lower: similarity,
      upper: similarity,
      count: group.length,
      value:
        group.reduce((sum, row) => sum + Math.abs(row.actual - row.predicted), 0) / group.length,
    };
  });
  if (card.prediction_kind === "probability") {
    card.classification_by_similarity = similarities.map((similarity) => {
      const group = rows.filter((row) => row.similarity === similarity);
      const tp = group.filter((r) => r.actual === 1 && r.predicted >= 0.3).length;
      const fn = group.filter((r) => r.actual === 1 && r.predicted < 0.3).length;
      const fp = group.filter((r) => r.actual === 0 && r.predicted >= 0.3).length;
      const tn = group.filter((r) => r.actual === 0 && r.predicted < 0.3).length;
      return {
        lower: similarity,
        upper: similarity,
        count: group.length,
        summary: {
          true_positive: tp,
          false_negative: fn,
          false_positive: fp,
          true_negative: tn,
          precision: tp + fp ? tp / (tp + fp) : null,
          recall: tp + fn ? tp / (tp + fn) : null,
          cutoff_inclusive: true,
        },
      };
    });
  }
}

async function results(page: Page, cards: ScorecardResponse[]) {
  await page.addInitScript(() => localStorage.setItem("theme", "light"));
  await installAuth(page);
  await installApiMock(page);
  await page.route("**/api/v1/engines**", (route) =>
    route.fulfill({
      json: [
        { id: "ecfp4-xgboost", name: "XGBoost" },
        { id: "ecfp4-randomforest", name: "Random forest" },
      ],
    }),
  );
  await page.route(`**/api/v1/protocols/${PROTOCOL_ID}`, (route) =>
    route.fulfill({
      json: {
        id: PROTOCOL_ID,
        name: "Solubility screening model",
        engine_id: "ecfp4-xgboost",
        status: "draft",
        conditions: {},
        can_delete: false,
        readouts: cards.map((card) => ({
          name: card.target,
          type: card.prediction_kind === "value" ? "numeric" : "class",
          unit: card.unit,
        })),
      },
    }),
  );
  await page.route(`**/api/v1/protocols/${PROTOCOL_ID}/scorecard`, (route) =>
    route.fulfill({ json: cards }),
  );
  await page.route(`**/api/v1/protocols/${PROTOCOL_ID}/scorecard/tolerance?**`, async (route) => {
    const params = new URL(route.request().url()).searchParams;
    const target = params.get("target");
    const tolerance = Number(params.get("tolerance"));
    const card = cards.find((item) => item.target === target);
    if (!card) throw new Error(`Unknown target: ${target}`);
    await route.fulfill({
      json: {
        target,
        tolerance,
        test_count: card.test_count,
        within_count: card.parity.filter(
          (point) => Math.abs(point.actual - point.predicted) <= tolerance,
        ).length,
        by_similarity: card.error_by_similarity.map((bin) => ({
          lower: bin.lower,
          upper: bin.upper,
          count: bin.count,
          within_count: card.parity.filter(
            (point) =>
              point.similarity != null &&
              point.similarity >= bin.lower &&
              point.similarity <= bin.upper &&
              Math.abs(point.actual - point.predicted) <= tolerance,
          ).length,
        })),
      },
    });
  });
  await signIn(page);
  await page.goto(`/protocols/${PROTOCOL_ID}`);
  await expect(
    page.getByRole("heading", { name: "How well does your model predict?" }),
  ).toBeVisible();
}

test.afterEach(async ({ page }) => clearBrowserState(page));

test("continuous results explain errors and calculate a custom tolerance per target", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1100 });
  await results(page, [regression, { ...regression, target: "pIC50", unit: "pIC50" }]);
  const overview = page.getByRole("region", { name: "Performance overview" });
  await expect(overview).toContainText("200 compounds");
  await expect(overview).toContainText("0.750log mol/L");
  await expect(overview).toContainText("0.780");
  await expect(overview).toContainText("25.0% lower");
  await expect(overview).toContainText("Set your tolerance");
  const input = page.getByRole("spinbutton", { name: /Acceptable error/ });
  await input.fill("0.5");
  await page.getByRole("button", { name: "Check", exact: true }).click();
  await expect(overview).toContainText("50.0%");
  await expect(overview).toContainText("100 of 200 test predictions");
  const similarity = page
    .locator('[data-slot="card"]')
    .filter({ hasText: "Are predictions better for familiar compounds?" });
  await expect(similarity.locator('svg[aria-label*="Average prediction error"]')).toHaveCount(1);
  await similarity.screenshot({ path: "test-results/scorecard-similarity-error.png" });
  await page.getByRole("button", { name: "Within acceptable error", exact: true }).click();
  await expect(page.getByText("Within acceptable error: higher is better")).toBeVisible();
  await expect(
    page.getByText(/100.0% \(25 of 25\) in the least-similar group and 0.0%/),
  ).toBeVisible();
  await similarity.screenshot({ path: "test-results/scorecard-similarity-tolerance.png" });
  await input.fill("1");
  await expect(overview).not.toContainText("50.0%");
  await expect(
    page.getByRole("button", { name: "Within acceptable error", exact: true }),
  ).toBeDisabled();
  await input.press("Enter");
  await expect(overview).toContainText("100.0%");
  await expect(
    page.getByText(/100.0% \(25 of 25\) in the least-similar group and 100.0%/),
  ).toBeVisible();
  const ranking = page.getByRole("region", { name: "Top-ranked test compounds" });
  await expect(ranking).toHaveCount(0);
  await expect(page.getByText("Largest prediction errors", { exact: true })).toHaveCount(0);
  await page.screenshot({ path: "test-results/scorecard-continuous-light.png", fullPage: true });
  await page.getByRole("tab", { name: "Test compounds", exact: true }).click();
  await expect(page.getByText("Largest prediction errors", { exact: true })).toBeVisible();
  await expect(ranking.getByRole("row").nth(1)).toContainText("ASSAY-200");
  await ranking.getByLabel("Ranking goal").selectOption("low");
  await expect(ranking.getByRole("row").nth(1)).toContainText("ASSAY-1");
  await ranking.screenshot({ path: "test-results/scorecard-ranking.png" });
  await page.getByRole("tab", { name: "pIC50", exact: true }).click();
  await expect(ranking.getByRole("row").nth(1)).toContainText("ASSAY-200");
  await page.getByRole("tab", { name: "Model performance", exact: true }).click();
  await expect(
    page.getByRole("spinbutton", { name: "Acceptable error (pIC50)", exact: true }),
  ).toHaveValue("");
  await expect(overview).toContainText("Set your tolerance");
  await expect(page.getByRole("tab", { name: "pIC50", exact: true })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(page.getByRole("heading", { name: "All metrics and comparisons" })).toBeVisible();
  await expect(page.getByText("Validation", { exact: false }).first()).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByRole("heading", { name: "Average prediction error", exact: true }),
  ).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  await page.getByRole("tab", { name: "Test compounds", exact: true }).click();
  await expect(ranking).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  await page.screenshot({ path: "test-results/scorecard-compounds-mobile.png" });
});

test("binary results show an accessible confusion matrix, precision, recall and the saved cutoff", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1100 });
  await results(page, [binary]);
  const overview = page.getByRole("region", { name: "Performance overview" });
  await expect(overview).toContainText("75.5%");
  await expect(overview).toContainText("80.0%");
  await expect(overview).toContainText("50 active · 150 inactive");
  const comparison = page.getByRole("region", { name: "Comparison & reliability" });
  const mcc = comparison.getByRole("region", { name: "MCC comparison" });
  const pr = comparison.getByRole("region", { name: "PR AUC comparison" });
  await expect(mcc).toContainText("0.700");
  await expect(mcc).toContainText("0.500");
  await expect(mcc).toContainText("+0.200");
  await expect(pr).toContainText("0.800");
  await expect(pr).toContainText("0.840");
  await expect(pr).toContainText("-0.040");
  await expect(pr).toContainText("50 of 200 test compounds are active (25.0%)");
  await expect(pr).not.toContainText("95% interval");
  await expect(comparison).toContainText("Outperforms the baseline (MCC)");
  await expect(comparison).toHaveClass(/border-success\/40/);
  await expect(mcc.getByText("Ahead of baseline", { exact: true })).toHaveAttribute(
    "data-variant",
    "success",
  );
  await expect(pr.getByText("Behind baseline", { exact: true })).toHaveAttribute(
    "data-variant",
    "warning",
  );
  await comparison.screenshot({ path: "test-results/scorecard-binary-comparison.png" });
  const matrix = page.getByRole("region", { name: "Confusion matrix" });
  await expect(matrix).toContainText("Decision cutoff: 0.3");
  await expect(matrix.getByRole("cell", { name: "40 Actives found" })).toBeVisible();
  await expect(matrix.getByRole("cell", { name: "10 Actives missed" })).toBeVisible();
  await expect(matrix.getByRole("cell", { name: "13 False alarms" })).toBeVisible();
  await expect(
    matrix.getByRole("cell", { name: "137 Inactives correctly rejected" }),
  ).toBeVisible();
  await expect(page.getByRole("spinbutton", { name: /Acceptable error/ })).toHaveCount(0);
  await page.getByRole("tab", { name: "Test compounds", exact: true }).click();
  const ranking = page.getByRole("region", { name: "Top-ranked test compounds" });
  await expect(ranking).toContainText("20 of 20 were actually active (100.0%)");
  await expect(ranking).toContainText("about 5.0 active compounds");
  await ranking.getByLabel("Ranking goal").selectOption("low");
  await expect(ranking).toContainText("10 of 20 were actually inactive (50.0%)");
  await expect(ranking).toContainText("about 15.0 inactive compounds");
  await page.getByRole("tab", { name: "Model performance", exact: true }).click();
  const rates = page.getByRole("table", { name: "Mistakes by similarity group" });
  await expect(rates).toContainText("33.3% (10 of 30)");
  await expect(rates).toContainText("26.0% (13 of 50)");
  await expect(rates).toContainText("Unavailable (no actives)");
  await expect(rates).toContainText("Unavailable (no inactives)");
  await expect(page.getByText("Mistake rate: lower is better")).toBeVisible();
  const similarity = page
    .locator('[data-slot="card"]')
    .filter({ hasText: "Are predictions better for familiar compounds?" });
  await expect(similarity.locator('svg[aria-label="Mistake rate: lower is better"]')).toHaveCount(
    1,
  );
  await similarity.screenshot({ path: "test-results/scorecard-similarity-binary.png" });
  await page.screenshot({ path: "test-results/scorecard-binary-light.png", fullPage: true });
  await page.evaluate(() => document.documentElement.setAttribute("data-theme", "dark"));
  await page.screenshot({ path: "test-results/scorecard-binary-dark.png", fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await comparison.screenshot({ path: "test-results/scorecard-binary-comparison-mobile.png" });
  await similarity.screenshot({ path: "test-results/scorecard-similarity-mobile.png" });
  await matrix.scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  await page.screenshot({ path: "test-results/scorecard-binary-mobile.png", fullPage: true });
});

test("ranking follows a saved low direction and asks when the direction is missing", async ({
  page,
}) => {
  await results(page, [
    { ...regression, direction: "low" },
    { ...regression, target: "unknown", direction: null },
  ]);
  await page.getByRole("tab", { name: "Test compounds", exact: true }).click();
  const ranking = page.getByRole("region", { name: "Top-ranked test compounds" });
  await expect(ranking.getByLabel("Ranking goal")).toHaveValue("low");
  await expect(ranking.getByRole("row").nth(1)).toContainText("ASSAY-1");
  await page.getByRole("tab", { name: "unknown", exact: true }).click();
  await expect(ranking).toContainText("This target has no saved preference");
  await expect(ranking.getByRole("table")).toHaveCount(0);
  await ranking.getByLabel("Ranking goal").selectOption("high");
  await expect(ranking.getByRole("row").nth(1)).toContainText("ASSAY-200");
});

test("small test sets show only available compounds and explain missing similarity data", async ({
  page,
}) => {
  await results(page, [
    {
      ...regression,
      test_count: 3,
      error_by_similarity: [],
      worst_rows: [],
      ranked_high: regression.ranked_high
        .slice(0, 3)
        .map((row, index) => ({ ...row, test_index: index, compound_id: null, similarity: null })),
      ranked_low: regression.ranked_low.slice(0, 3),
    },
  ]);
  const ranking = page.getByRole("region", { name: "Top-ranked test compounds" });
  await expect(
    page.getByText(
      "There are not enough test compounds with training-similarity data to show a comparison.",
    ),
  ).toBeVisible();
  await page.getByRole("tab", { name: "Test compounds", exact: true }).click();
  await expect(ranking).toContainText("Top 3 of 3 test compounds");
  await expect(ranking.getByRole("row")).toHaveCount(4);
  await expect(ranking.getByRole("row").nth(1)).toContainText("Test compound 1");
  await expect(ranking.getByRole("row").nth(1)).toContainText("Unavailable");
});
