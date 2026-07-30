# daikon-studio — Phase 2 UX pass, design

**Date:** 2026-07-29
**Status:** approved in brainstorming; awaiting implementation plan
**Predecessors:** `2026-07-29-daikon-studio-frontend-design.md` (the Phase 1 UI and its
conventions — all still binding), `2026-07-25-daikon-studio-phase-1-design.md` (the
product; §2's terminology is load-bearing).

## 1. What this is

The polish pass the Phase 1 handoff named as the intended next step, plus the one
backend gap that blocks it: server-side sorting and filtering of run results. Five
pieces, chosen and shaped with the user:

1. The results API gains sort, range filters, and — load-bearing — a stable per-row
   identity.
2. The triage grid gains native AG Grid column filters and an "In domain only" switch.
3. The Scorecard becomes a single verdict strip: verdict and honesty numbers in one
   border, "Where it fails" promoted above the metric table.
4. The predict flow becomes an enriched single page: preview before Run, protocol
   context with read-only trained-with conditions, an honest progress card, and an
   explicit cache-hit acknowledgment.
5. Structures render theme-aware instead of `dark:invert`.

Plus a live-verification pass over the two paths no browser has seen: a binary
classification protocol, and `baseline_is_self`.

Everything builds on what is already installed — shadcn/ui, AG Grid, RDKit.js,
papaparse, react-dropzone. The only addition is the shadcn `Progress` component,
which replaces two hand-rolled `<div>` bars.

**Out of scope:** SMILES/SMARTS search, a custom filter bar, the dashboard,
Playwright, Phase 2 product features (Sweep, lineage, Sources, Proposals,
generation), and the Sentinel `create_service_app` one-liner (standing rule: no
identity-service changes without explicit in-the-moment consent).

## 2. Results API: sort, filters, row identity

`GET /runs/{run_id}/results` gains three optional query parameters:

- `sort_by`: a readout name, `uncertainty`, or `applicability`.
- `sort_dir`: `asc` | `desc`, default `asc`. Single-column sort only — AG Grid
  multi-sort is not wired, and nothing asked for it.
- `filters`: URL-encoded JSON object, `{"<column>": {"min": <number>?, "max":
  <number>?}}`, over the same column vocabulary. At least one bound per entry.

The "In domain only" toggle is client sugar over `{"applicability": {"min": 0.5}}`;
no dedicated parameter exists for it.

**`PredictionResponse` gains `row_id: int`** — the row's index in the original
Parquet, minted with `.with_row_index()` *before* any filter or sort. This is the
correctness core of the whole feature: today the client derives row identity from
pagination offset (`startRow + index`), and under server-side reordering that would
silently save the wrong compounds into a Collection. `POST /collections` is
unchanged — `row_id` is exactly the original-file offset its `row_ids` already
expect.

Mechanics, in `GetPredictionResults`:

- The frame is already read fully into memory per request (existing, ponytail-marked),
  so filter and sort are cheap Polars operations applied before the existing
  slice: `with_row_index` → filter → sort → slice.
- Nulls sort last regardless of direction (`nulls_last=True`); a range filter on a
  null value excludes the row. `uncertainty` is null for XGBoost and
  `applicability` can be null, so both cases are real.
- The cursor stays a plain integer offset and now means "offset within this
  filtered and sorted view". The results file is immutable, and the client resets
  to offset 0 whenever sort or filters change (AG Grid purges its block cache on
  either), so the offset still has nothing to race.
- `GetPredictionResultsQuery` gains typed fields — `sort: SortSpec | None`,
  `filters: tuple[RangeFilter, ...]` (frozen dataclasses beside the query). The
  route parses the wire JSON into them; the use case validates column names
  against the protocol's readouts plus the two fixed columns, because what counts
  as a valid column is domain knowledge. Violations return 422
  (`ValidationError`), same envelope as the existing cursor error.

The OpenAPI snapshot (`make` target that writes `frontend/openapi.json`) and the
orval-generated types (`pnpm generate:api`) are regenerated; hooks stay
hand-written per convention.

Backend tests: each sort direction; min-only, max-only and both-bound filters;
sort and filter combined; unknown column and malformed JSON → 422; null ordering;
`row_id` stability under filter+sort (the same compound keeps the same `row_id`
in every view); pagination across a filtered view (last-page detection still via
the fetch-one-extra-row mechanism).

## 3. Triage grid: native column filters, in-domain switch

- Numeric columns — one per readout, plus uncertainty and applicability — become
  `sortable: true` with `agNumberColumnFilter`, `filterParams` restricted to the
  three operations the API can honour: greater-than-or-equal, less-than-or-equal,
  in-range. Structure and SMILES columns stay unsortable and unfilterable; text
  search is out of scope, and a control that silently does nothing stays banned.
- The datasource translates AG Grid's `sortModel`/`filterModel` into API
  parameters through a pure function in `features/runs/lib`, unit-tested. It also
  owns the merge rule for the one composition case: the "In domain only" switch
  and a user's own applicability column filter intersect (max of mins, min of
  maxes).
- The "In domain only" control is a shadcn `Switch` above the grid, left of the
  selection summary. Flipping it purges the grid's block cache (same path as any
  filter change).
- `TriageRow.__rowId` is now populated from the API's `row_id` instead of being
  computed. `getRowId` is unchanged, so selections survive filter and sort
  changes, and "Save N as collection" saves the rows the chemist actually saw.
- The comment block explaining why the grid has no filter box is deleted along
  with the reason for it.

## 4. Scorecard: the verdict strip

The verdict band and the three honesty cards fuse into one bordered unit, so the
verdict cannot be read without its caveats — the anti-flattery rule enforced by
layout rather than discipline:

- Line 1: the verdict headline (unchanged copy, unchanged `computeVerdict`).
- Line 2: primary metric — model, baseline, margin (unchanged).
- Line 3, inside the same border, separated by a hairline: **optimism gap**,
  **assay noise floor**, **applicability coverage**, each a number with a
  one-line caption. The captions stay visible text — no tooltips; the page's job
  is to be uncomfortable, and hover-to-see-the-caveat would undo it. The
  `within-noise` explanatory sentence stays in the band as today.
- Absence rules carry over exactly: noise floor is *absent, not empty* for a
  binary target; applicability shows its "could not be computed" state; the
  `is-baseline` and `unknown` verdicts keep their current prose inside the strip.
- The three-card grid (`OptimismGapCard`, the two `HonestyCard` usages) is
  deleted.

Page order becomes **verdict strip → Where it fails → All metrics**. ESOL proved
the failure view carries the actionable finding (8 of 20 misses on "no ring
system"); the full table is reference material. The table itself is unchanged and
stays fully expanded — collapsing it would be a hiding mechanism.

The applicability meter in the strip and the run-detail progress bar both become
the shadcn `Progress` component (added via the shadcn CLI).

`lib/verdict.ts` and its tests are untouched; this is a layout change.

## 5. Predict flow: enriched single page

The wizard stays one page; sections appear as the user progresses.

- **Protocol context card**, shown once a protocol is chosen: what it predicts
  (readout name, unit, direction), the engine, version and trained date, and the
  conditions it was trained with — **read-only**, values from
  `protocol.conditions` labelled via the engine manifest. Read-only is a checked
  fact, not a shortcut: both engines ignore `ctx.conditions` at predict time
  (`_predict_with_tree_ensemble` never reads them), so editable fields would be
  dead controls. If an engine ever declares predict-time conditions, inputs earn
  their place then. If an honest training-compound count is available from the
  dataset without new backend work, show it; otherwise omit it — never
  `valid_rows`, which is not a compound count.
- **Preview on drop.** papaparse parses the whole file (no more `preview: 5`).
  The panel shows: the compound count; the structure-column select (existing);
  the first 6 `StructureThumbnail`s from the chosen column, re-rendered when the
  column changes; and an RDKit parse check over the first 100 rows — "3 of the
  first 100 didn't parse", with copy about what happens to those rows written to
  match what the prediction worker actually does (verify in code first — the
  dataset-upload skip semantics do not automatically apply).
- **The Run button states the commitment:** "Score 120 compounds" — an
  operate-on count, so it sits inside the button per the counts convention.
- **Post-submit:** navigate to `/runs/{id}?compounds={n}`, plus `&cached=1` when
  the 202 response arrives already `ready` (the cache-hit trap). Run detail:
  - With `compounds`: "Scoring 120 compounds · submitted 14:32" above the
    progress bar. Without it (a run opened from the list): no line — absence
    renders as absence, and no backend change is spent on a nicety.
  - With `cached=1` and a ready run: a dismissible note — "These compounds were
    already scored by this protocol — results are from cache" — instead of a
    progress bar that never had work to show.
  - The progress card uses `Progress` plus the existing phase copy.

## 6. Dark-mode structures

`StructureThumbnail` drops `dark:invert` (which puts structures on black
rectangles and turns heteroatom colours into complements) and renders theme-aware:

- Background transparent in both themes.
- Dark mode passes a light bond/atom palette through RDKit's drawing options;
  light mode keeps the default palette.
- Theme comes from the existing next-themes provider; theme is a dependency of
  the render effect, so switching themes re-renders thumbnails.
- Exact rdkit-js drawing-option names (`backgroundColour`, palette keys) get
  verified against the installed rdkit-js version during implementation, not
  assumed.

## 7. Live verification: the two unexercised paths

After the Scorecard rework lands, drive both in a real browser against the live
backend (SACLAB-DEV workspace, staged MoleculeNet files):

1. **Binary classification end to end** — `bace-active.csv`, XGBoost: MCC and
   balanced accuracy lead the metric table; the noise-floor stat is *absent* from
   the verdict strip, not rendered empty.
2. **`baseline_is_self`** — ECFP4+RandomForest on the same data: the strip says
   "this model is the baseline"; no comparison is rendered anywhere.

Both have unit tests; neither has been seen live. Both exercise the strip's
absent-states, which is why they gate this pass rather than trail it.

## 8. Acceptance

- Backend: new tests from §2 pass; full suite stays green (285 at last count).
- Frontend: translate/merge functions unit-tested; existing verdict tests green;
  `tsc` and `biome` clean **by exit code**.
- The Phase 1 loop still runs end to end in a browser, plus the two §7 paths.
- Conventions from the Phase 1 design doc hold: `ReadoutValue` for every number,
  absence as absence, no UUIDs surfaced, counts placed by semantics, hand-curated
  barrels, orval types only.

## 9. Points to verify during planning/implementation

Recorded so the plan checks them instead of inheriting guesses:

1. What the prediction worker does with an unparseable structure (drops the row?
   fails the run?) — the preview copy in §5 depends on it.
2. rdkit-js drawing-option names for background and palette on the installed
   version.
3. Whether `DatasetResponse` exposes an honest post-grouping compound count for
   the context card (`valid_rows` is disqualified).
4. That AG Grid's infinite model resets to offset 0 on filter/sort model changes
   with the community build in use (it should purge the block cache; confirm, or
   purge explicitly in the datasource).
