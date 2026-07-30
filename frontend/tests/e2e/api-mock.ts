import type { Page, Request } from "@playwright/test";

/**
 * A stand-in for `/api/v1/**`.
 *
 * The fixture is built so that a page's row order and its `row_id`s **never**
 * agree. That is the whole point: the bug this suite exists to catch is a
 * client that derives row identity from pagination position (`startRow +
 * index`) instead of reading the server's `row_id`. Against a fixture whose
 * ids happen to be `0,1,2,3`, both implementations look identical and the test
 * proves nothing.
 */

export const RUN_ID = "11111111-1111-4111-8111-111111111111";
export const PROTOCOL_ID = "22222222-2222-4222-8222-222222222222";
export const READOUT = "log_solubility";

interface ResultRow {
  row_id: number;
  structure: string;
  value: number;
  applicability: number;
}

/**
 * Six compounds whose `row_id`s are deliberately scattered. The SMILES are
 * distinct and short so a row can be identified on screen by its text.
 *
 * **No row's id may equal its index**, in this unsorted order or after any
 * sort a test performs. Where the two coincide, an offset-derived id and a
 * server-supplied one produce the same number and the assertion goes blind —
 * that is not hypothetical, an earlier draft of this fixture had `CCF` at both
 * index 5 and id 5, and the filtering test below silently stopped catching the
 * bug. `assertIdsNeverMatchIndex` guards the unsorted case.
 */
export const RESULT_ROWS: ResultRow[] = [
  { row_id: 7, structure: "CCO", value: -0.77, applicability: 0.95 },
  { row_id: 3, structure: "CCN", value: -1.12, applicability: 0.88 },
  { row_id: 11, structure: "CCCO", value: -1.55, applicability: 0.72 },
  { row_id: 8, structure: "CCBr", value: -2.31, applicability: 0.64 },
  { row_id: 9, structure: "CCCl", value: -3.04, applicability: 0.51 },
  { row_id: 2, structure: "CCF", value: -4.2, applicability: 0.22 },
];

/** Fails loudly if the fixture ever drifts into the blind spot described above. */
export function assertIdsNeverMatchIndex(): void {
  const collisions = RESULT_ROWS.map((row, index) => ({ row, index })).filter(
    ({ row, index }) => row.row_id === index,
  );
  if (collisions.length > 0) {
    throw new Error(
      `Fixture row_id collides with its index for: ${collisions
        .map(({ row, index }) => `${row.structure} (index ${index}, row_id ${row.row_id})`)
        .join(", ")}. An offset-derived id would be indistinguishable from the real one.`,
    );
  }
}

const PROTOCOL = {
  id: PROTOCOL_ID,
  workspace_id: "00000000-0000-4000-8000-000000000001",
  name: "E2E Solubility",
  dataset_id: "33333333-3333-4333-8333-333333333333",
  engine_id: "ecfp4-xgboost",
  artifact_uri: "file:///dev/null",
  readouts: [{ name: READOUT, unit: "log mol/L", direction: "high" }],
  conditions: {},
  status: "published",
  is_locked: true,
  published_at: "2026-07-30T00:00:00Z",
  parent_protocol_id: null,
  protocol_version: 1,
  created_at: "2026-07-30T00:00:00Z",
};

const RUN = {
  id: RUN_ID,
  workspace_id: "00000000-0000-4000-8000-000000000001",
  kind: "prediction",
  status: "ready",
  progress: 1,
  phase: null,
  result_uri: "file:///dev/null",
  error_message: null,
  protocol_id: PROTOCOL_ID,
  created_at: "2026-07-30T00:00:00Z",
};

/** Every `POST /collections` body the app sent, in order. */
export type CollectionPosts = Array<{ name: string; run_id: string; row_ids: number[] }>;

function json(body: unknown, status = 200) {
  return { status, contentType: "application/json", body: JSON.stringify(body) };
}

/**
 * Apply the server's own sort/filter semantics to the fixture.
 *
 * Mirroring them here is what lets the test assert that the rows it ticked are
 * the rows whose ids get posted: the grid shows whatever this returns, so the
 * expected ids come from the same function the page is driven by.
 */
function view(request: Request): ResultRow[] {
  const url = new URL(request.url());
  let rows = [...RESULT_ROWS];

  const raw = url.searchParams.get("filters");
  if (raw) {
    const filters = JSON.parse(raw) as Record<string, { min?: number; max?: number }>;
    for (const [column, bounds] of Object.entries(filters)) {
      rows = rows.filter((row) => {
        const value = column === READOUT ? row.value : row.applicability;
        if (bounds.min != null && value < bounds.min) return false;
        if (bounds.max != null && value > bounds.max) return false;
        return true;
      });
    }
  }

  const sortBy = url.searchParams.get("sort_by");
  if (sortBy) {
    const descending = url.searchParams.get("sort_dir") === "desc";
    const pick = (row: ResultRow) => (sortBy === READOUT ? row.value : row.applicability);
    // Ties break on row_id, exactly as `result_view.py` does, so a page is a
    // stable window here for the same reason it is server-side.
    rows.sort((a, b) => pick(a) - pick(b) || a.row_id - b.row_id);
    if (descending) rows.reverse();
  }

  return rows;
}

/**
 * Serve `/api/v1/**` from the fixture, and record what the app posts back.
 *
 * The returned array is the assertion surface: the test reads it to see which
 * `row_ids` the grid actually sent when the user clicked save.
 */
export async function installApiMock(page: Page): Promise<CollectionPosts> {
  const collectionPosts: CollectionPosts = [];

  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const { pathname, searchParams } = new URL(request.url());

    if (request.method() === "POST" && pathname.endsWith("/api/v1/collections")) {
      collectionPosts.push(JSON.parse(request.postData() ?? "{}"));
      await route.fulfill(
        json({ id: "44444444-4444-4444-8444-444444444444", member_count: 0 }, 201),
      );
      return;
    }

    if (pathname.endsWith(`/runs/${RUN_ID}/results`)) {
      const rows = view(request);
      const offset = Number(searchParams.get("cursor") ?? 0);
      const limit = Number(searchParams.get("limit") ?? 100);
      const page_ = rows.slice(offset, offset + limit);
      await route.fulfill(
        json({
          items: page_.map((row) => ({
            row_id: row.row_id,
            structure: row.structure,
            readouts: { [READOUT]: { value: row.value, unit: "log mol/L", direction: "high" } },
            uncertainty: null,
            applicability: row.applicability,
          })),
          next_cursor: offset + limit < rows.length ? String(offset + limit) : null,
          total_count: null,
        }),
      );
      return;
    }

    if (pathname.endsWith(`/runs/${RUN_ID}`)) {
      await route.fulfill(json(RUN));
      return;
    }
    if (pathname.endsWith(`/protocols/${PROTOCOL_ID}`)) {
      await route.fulfill(json(PROTOCOL));
      return;
    }
    if (pathname.endsWith("/api/v1/protocols")) {
      await route.fulfill(json({ items: [PROTOCOL], next_cursor: null, total_count: null }));
      return;
    }

    // Anything unmodelled returns an empty page rather than falling through to
    // the network — see playwright.config.ts for why nothing may reach a real
    // host from these tests.
    await route.fulfill(json({ items: [], next_cursor: null, total_count: null }));
  });

  return collectionPosts;
}
