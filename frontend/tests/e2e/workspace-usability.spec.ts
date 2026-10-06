import { expect, test } from "@playwright/test";
import { installApiMock } from "./api-mock";
import { clearBrowserState, installAuth, signIn } from "./auth";

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem("theme", "light"));
  await installAuth(page);
  await installApiMock(page);
});

test.afterEach(async ({ page }) => {
  await clearBrowserState(page);
});

test("mobile navigation opens, closes after selection, and exposes quick actions", async ({
  page,
}) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await signIn(page);

  await page.getByRole("button", { name: "Open navigation" }).click();
  const navigation = page.getByRole("dialog", { name: "Sidebar" });
  await expect(navigation).toBeVisible();
  await navigation.getByRole("link", { name: "Datasets", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Datasets", exact: true })).toBeVisible();
  await expect(navigation).toBeHidden();

  // Font scaling and a small viewport must not push navigation or actions offscreen.
  await page.evaluate(() => {
    document.documentElement.style.fontSize = "120%";
  });
  await expect(page.getByRole("button", { name: "Open navigation" })).toBeInViewport();
  await expect(page.getByRole("button", { name: "Go to a page or action" })).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  await page.screenshot({ path: "test-results/workspace-mobile.png", fullPage: true });

  await page.getByRole("button", { name: "Go to a page or action" }).click();
  await page.getByRole("option", { name: "Upload dataset", exact: true }).click();
  await expect(page).toHaveURL(/\/datasets\/new$/);
  await expect(page.getByRole("heading", { name: "New dataset" })).toBeVisible();
  await expect(page.getByRole("dialog")).toBeHidden();
});

test("a failed dataset request can be retried without leaving the selected folder", async ({
  page,
}) => {
  let fail = true;
  const folders: (string | null)[] = [];
  await page.route("**/api/v1/datasets?**", async (route) => {
    const folder = new URL(route.request().url()).searchParams.get("folder_id");
    if (!folder) return route.fallback();
    folders.push(folder);
    await route.fulfill({
      status: fail ? 503 : 200,
      contentType: "application/json",
      body: JSON.stringify(
        fail ? { detail: "Temporarily unavailable" } : { items: [], next_cursor: null },
      ),
    });
  });
  await signIn(page);
  await page.goto("/datasets?folder=research");
  const error = page.getByRole("alert").filter({ hasText: "Could not load datasets" });
  await expect(error).toBeVisible();
  fail = false;
  await error.getByRole("button", { name: "Try again" }).click();
  await expect(error).toBeHidden();
  await expect(
    page.getByText("This folder is empty. Drag a dataset here or use Move to folder."),
  ).toBeVisible();
  await expect(page).toHaveURL(/\/datasets\?folder=research$/);
  expect(folders.length).toBeGreaterThanOrEqual(2);
  expect(folders.every((folder) => folder === "research")).toBe(true);
});

test("compact cards preview real structures and work in both themes", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  let sampleRequests = 0;
  await page.route("**/api/v1/datasets**", async (route) => {
    if (new URL(route.request().url()).pathname.endsWith("/compounds")) {
      sampleRequests++;
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          total: 1128,
          items: ["CC(=O)Oc1ccccc1C(=O)O", "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)Nc1ccc(O)cc1"].map(
            (structure, index) => ({
              structure,
              compound_id: `ESOL-${index + 1}`,
              targets: { logS: [-2.18, -0.77, -1.42][index] },
              split: "train",
            }),
          ),
        }),
      });
      return;
    }
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        items: ["Solubility reference", "Kinase activity", "Permeability screen"].map(
          (name, index) => ({
            id: `dataset-${index}`,
            name,
            targets: [{ column: index === 1 ? "pIC50" : "logS", kind: "numeric" }],
            row_count: [1128, 2460, 836][index],
            split: { strategy: "scaffold", seed: 42 },
            created_at: "2026-10-05T12:00:00Z",
            created_by: null,
            folder_id: null,
          }),
        ),
        next_cursor: null,
      }),
    });
  });
  await signIn(page);
  await expect(page.getByRole("heading", { name: "Dashboard", exact: true })).toBeVisible();
  await expect(page.getByRole("link", { name: "Solubility reference" })).toBeVisible();
  expect(
    await page
      .getByRole("heading", { name: "Dashboard", exact: true })
      .evaluate((heading) => getComputedStyle(heading).fontFamily),
  ).toMatch(/Inter/i);
  await page.screenshot({ path: "test-results/dashboard-home.png", fullPage: true });
  await page.goto("/datasets");
  await expect(page.getByRole("link", { name: "Solubility reference" })).toBeVisible();
  await expect(page.getByRole("link", { name: "New dataset" })).toBeInViewport();
  await expect(page.getByRole("main")).toHaveCount(1);
  expect(sampleRequests).toBe(0);
  await page.getByRole("link", { name: "Solubility reference" }).hover();
  await page.getByRole("button", { name: "Preview structures in Solubility reference" }).click();
  const sample = page.getByRole("region", { name: "Structure preview for Solubility reference" });
  await expect(sample.getByRole("img")).toHaveCount(3);
  await expect(sample.getByText("-2.180")).toBeVisible();
  expect(sampleRequests).toBe(1);
  await page.screenshot({ path: "test-results/dataset-preview.png", fullPage: true });
  await page.getByRole("dialog").getByRole("button", { name: "Close", exact: true }).click();
  await expect(sample).toBeHidden();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.screenshot({ path: "test-results/workspace-desktop-light.png", fullPage: true });
  await page.getByRole("button", { name: "Toggle theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.screenshot({ path: "test-results/workspace-desktop-dark.png", fullPage: true });
});

test("the dashboard shows active training and prediction work without inventing totals", async ({
  page,
}) => {
  let requestedAllKinds = false;
  await page.route("**/api/v1/runs?**", async (route) => {
    const params = new URL(route.request().url()).searchParams;
    requestedAllKinds = !params.has("kind") && params.getAll("status").includes("running");
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        items: [
          {
            id: "training-1",
            kind: "training",
            name: "Solubility model",
            status: "running",
            phase: "Fitting the model",
            progress: 0.4,
          },
          {
            id: "prediction-1",
            kind: "prediction",
            name: "Screening batch",
            status: "pending",
            phase: null,
            progress: 0,
          },
          {
            id: "done-1",
            kind: "prediction",
            name: "Finished batch",
            status: "ready",
            phase: null,
            progress: 1,
          },
        ],
        next_cursor: null,
      }),
    });
  });
  await signIn(page);
  await expect(page.getByRole("link", { name: /Solubility model/ })).toHaveAttribute(
    "href",
    "/runs/training-1",
  );
  await expect(page.getByRole("link", { name: /Screening batch/ })).toHaveAttribute(
    "href",
    "/runs/prediction-1",
  );
  await expect(page.getByText("Finished batch")).toHaveCount(0);
  expect(requestedAllKinds).toBe(true);
});

test("the font setting applies to all text and survives reload", async ({ page }) => {
  await signIn(page);
  await page.goto("/settings");

  for (const font of ["Merriweather", "IBM Plex", "Inter"]) {
    const choice = page.getByRole("button", { name: font, exact: true });
    await choice.click();
    await expect(choice).toHaveAttribute("aria-pressed", "true");
    await expect
      .poll(() =>
        page.evaluate(() => {
          const bodyFont = getComputedStyle(document.body).fontFamily;
          const elements = document.querySelectorAll(
            "h1, h2, h3, p, button, input, [class~='font-mono'], [class~='font-sans']",
          );
          return (
            Array.from(elements).every(
              (element) => getComputedStyle(element).fontFamily === bodyFont,
            ) &&
            bodyFont
              .toLowerCase()
              .includes(
                document.documentElement.dataset.font === "plex"
                  ? "plex"
                  : (document.documentElement.dataset.font ?? ""),
              )
          );
        }),
      )
      .toBe(true);
    await page.reload();
    await expect(page.getByRole("button", { name: font, exact: true })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  }
});
