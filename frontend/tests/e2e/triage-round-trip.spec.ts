import { expect, test } from "@playwright/test";
import { RESULT_ROWS, RUN_ID, assertIdsNeverMatchIndex, installApiMock } from "./api-mock";
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
  await expect(page.getByText(RESULT_ROWS[0].structure, { exact: true })).toBeVisible();

  // Tick two rows by their structures, so the expectation is written in terms
  // of what a person would have seen and clicked.
  const picked = ["CCN", "CCCO"];
  for (const smiles of picked) {
    await page.getByRole("row", { name: new RegExp(`\\b${smiles}\\b`) }).getByRole("checkbox").check();
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
  await expect(page.getByText("CCF", { exact: true })).toBeVisible();

  // CCF sits at applicability 0.22, below the in-domain floor.
  await page.getByRole("row", { name: /\bCCF\b/ }).getByRole("checkbox").check();

  // Turning the switch on refetches with an applicability floor, which drops
  // the ticked row from the view entirely.
  await page.getByLabel(/within applicability domain/i).click();
  await expect(page.getByText("CCF", { exact: true })).toBeHidden();

  await page.getByRole("button", { name: /save 1 as collection/i }).click();
  await page.getByLabel(/name/i).fill("e2e filtered selection");
  await page.getByRole("button", { name: /^save collection$/i }).click();

  await expect.poll(() => collectionPosts.length).toBe(1);
  // AG Grid discards blocks it has scrolled or filtered past; the id is what
  // keeps a selection meaningful across that, which is the other half of why
  // it comes from the server.
  expect(collectionPosts[0].row_ids).toEqual([2]);
});
