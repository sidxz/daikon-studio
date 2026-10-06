import { expect, test } from "@playwright/test";
import {
  PROTOCOL_ID,
  RESULT_ROWS,
  RUN_ID,
  assertIdsNeverMatchIndex,
  installApiMock,
} from "./api-mock";
import { clearBrowserState, installAuth, signIn } from "./auth";

/**
 * The triage round trip, browser side.
 *
 * A chemist sorts and filters the grid, ticks rows, and saves them as a
 * Collection. What must travel with that click is each row's `row_id` — its
 * position in the *original* results file — never its position on the page.
 * Get that wrong and the Collection holds different compounds than the ones on
 * screen, with no error to notice.
 *
 * The fixture's `row_id`s are scattered (`7, 3, 11, 8, 9, 2`) and none of them
 * equals its own index, so a client that sends `startRow + index` sends
 * demonstrably different numbers and these assertions fail loudly. Against ids
 * that happened to be `0,1,2,3` the two implementations would be
 * indistinguishable and this file would be theatre — which is why
 * `assertIdsNeverMatchIndex` runs before every test.
 *
 * The backend half of the same round trip — that those ids still select the
 * right rows out of the Parquet after a sort — is
 * `backend/tests/api/test_triage_round_trip.py`.
 */

test.beforeEach(async ({ page }) => {
  // Guards the fixture against drifting into a shape where an offset and a
  // row_id coincide, which would leave these assertions passing for the wrong
  // reason.
  assertIdsNeverMatchIndex();
  await installAuth(page);
});

test.afterEach(async ({ page }) => {
  await clearBrowserState(page);
});

test("saving a selection posts the server's row_ids, not page offsets", async ({ page }) => {
  const collectionPosts = await installApiMock(page);
  await signIn(page);

  await page.goto(`/runs/${RUN_ID}`);

  // Sort descending on the readout so the page order stops matching file
  // order — the condition under which offset-derived ids go wrong.
  await page.getByRole("columnheader", { name: /log_solubility/ }).click();
  await page.getByRole("columnheader", { name: /log_solubility/ }).click();
  await expect(
    page.getByRole("img", { name: RESULT_ROWS[0].structure, exact: true }),
  ).toBeVisible();

  // Tick two rows by their structures, so the expectation is written in terms
  // of what a person would have seen and clicked.
  const picked = ["CCN", "CCCO"];
  for (const smiles of picked) {
    await page
      .getByRole("row")
      .filter({ has: page.getByRole("img", { name: smiles, exact: true }) })
      .getByRole("checkbox")
      .check();
  }

  await page.getByRole("button", { name: /save 2 as collection/i }).click();
  await page.getByLabel(/name/i).fill("e2e round trip");
  await page.getByRole("button", { name: /^save collection$/i }).click();

  await expect.poll(() => collectionPosts.length).toBe(1);

  const expected = RESULT_ROWS.filter((row) => picked.includes(row.structure)).map(
    (row) => row.row_id,
  );
  expect(collectionPosts[0].run_id).toBe(RUN_ID);
  expect([...collectionPosts[0].row_ids].sort((a, b) => a - b)).toEqual(
    [...expected].sort((a, b) => a - b),
  );

  // The ids the fixture assigns to those two compounds are 3 and 11. A client
  // deriving identity from position would have sent 1 and 2 here.
  expect(collectionPosts[0].row_ids).not.toContain(1);
  expect(collectionPosts[0].row_ids).not.toContain(2);
});

test("a selection survives filtering the rows out from under it", async ({ page }) => {
  const collectionPosts = await installApiMock(page);
  await signIn(page);

  await page.goto(`/runs/${RUN_ID}`);
  await page.getByRole("button", { name: "Maximize", exact: true }).click();
  await expect(page.getByRole("img", { name: "CCF", exact: true })).toBeVisible();

  // CCF sits at applicability 0.22, below the in-domain floor.
  await page
    .getByRole("row")
    .filter({ has: page.getByRole("img", { name: "CCF", exact: true }) })
    .getByRole("checkbox")
    .check();

  // Turning the switch on refetches with an applicability floor, which drops
  // the ticked row from the view entirely.
  await page.getByLabel(/within applicability domain/i).click();
  await expect(page.getByRole("img", { name: "CCF", exact: true })).toBeHidden();

  await page.getByRole("button", { name: /save 1 as collection/i }).click();
  await page.getByLabel(/name/i).fill("e2e filtered selection");
  await page.getByRole("button", { name: /^save collection$/i }).click();

  await expect.poll(() => collectionPosts.length).toBe(1);
  // AG Grid discards blocks it has scrolled or filtered past; the id is what
  // keeps a selection meaningful across that, which is the other half of why
  // it comes from the server.
  expect(collectionPosts[0].row_ids).toEqual([2]);
});

test("class filtering selects positive, negative, and all rows without repeated requests", async ({
  page,
}) => {
  await installApiMock(page);
  await page.route(`**/api/v1/protocols/${PROTOCOL_ID}`, async (route) => {
    // Serve the classification protocol entirely from fixtures.
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        id: PROTOCOL_ID,
        name: "Classification",
        engine_id: "ecfp4-xgboost",
        status: "published",
        readouts: [
          { name: "active", type: "class", threshold: 0.5 },
          { name: "reactive", type: "class", threshold: 0.5 },
        ],
        conditions: {},
      }),
    });
  });
  const queries: Array<string | null> = [];
  await page.route(`**/api/v1/runs/${RUN_ID}/results?**`, async (route) => {
    const raw = new URL(route.request().url()).searchParams.get("filters");
    queries.push(raw);
    const filters = raw ? JSON.parse(raw) : {};
    // Match the real API: grid-generated suffixes are not readout names.
    if (Object.keys(filters).some((column) => !["active", "reactive"].includes(column))) {
      await route.fulfill({
        status: 422,
        contentType: "application/json",
        body: JSON.stringify({ message: "Unknown readout column" }),
      });
      return;
    }
    const bounds = filters.active;
    const items = [0, 1]
      .filter((value) => !bounds || value === bounds.min)
      .map((value) => ({
        row_id: value + 7,
        compound_id: value ? "positive-compound" : "negative-compound",
        structure: value ? "CCN" : "CCO",
        readouts: {
          active: { value, type: "class" },
          reactive: { value: 1 - value, type: "class" },
        },
        uncertainty: {},
        applicability: 0.9,
      }));
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        items,
        next_cursor: null,
        total_count: items.length,
      }),
    });
  });
  await signIn(page);
  await page.goto(`/runs/${RUN_ID}`);
  await expect(page.getByText("positive-compound", { exact: true })).toBeVisible();
  const header = page.locator(".ag-header-cell[col-id='active']");
  await header.locator(".ag-header-cell-filter-button").click();
  await page.getByRole("radio", { name: "Positive", exact: true }).check();
  await expect(page.getByText("negative-compound", { exact: true })).toBeHidden();
  await expect(page.getByText("positive-compound", { exact: true })).toBeVisible();
  expect(JSON.parse(queries.at(-1) ?? "{}")).toEqual({
    active: { min: 1, max: 1 },
  });
  const count = queries.length;
  await page.waitForTimeout(500);
  expect(queries.length).toBe(count);
  if (!(await page.getByRole("radio", { name: "Negative", exact: true }).isVisible()))
    await header.locator(".ag-header-cell-filter-button").click();
  await page.getByRole("radio", { name: "Negative", exact: true }).check();
  await expect(page.getByText("positive-compound", { exact: true })).toBeHidden();
  await expect(page.getByText("negative-compound", { exact: true })).toBeVisible();
  if (!(await page.getByRole("radio", { name: "All classes", exact: true }).isVisible()))
    await header.locator(".ag-header-cell-filter-button").click();
  await page.getByRole("radio", { name: "All classes", exact: true }).check();
  await expect(page.getByText("positive-compound", { exact: true })).toBeVisible();
  await expect(page.getByText("negative-compound", { exact: true })).toBeVisible();
});
