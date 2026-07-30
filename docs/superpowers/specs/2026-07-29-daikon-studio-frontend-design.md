# daikon-studio — Phase 1 Frontend Design

**Date:** 2026-07-29
**Status:** Approved
**Scope:** The Phase 1 UI, plus the three backend gaps that block it.
**Predecessor:** `2026-07-25-daikon-studio-phase-1-design.md` (the product) and
`../HANDOFF-frontend.md` (the state of the world at the end of the backend build).

## 1. What this is

The Phase 1 backend is complete and merged to `main`: 19 endpoints, 272 tests, mypy strict,
three import-linter contracts. This spec covers the UI that walks the whole loop — upload a
CSV, freeze a Dataset, train a Protocol, read its Scorecard, publish it, run it over a
compound set, triage the results, export a Collection — and the small amount of backend work
that has to land first.

The goal is the entire loop functioning with plain, suite-standard styling. Visual refinement
is a later pass and is deliberately not specified here.

### The claim the UI has to keep

The product exists because no-code ML platforms produce models that score beautifully on a
random split and fail in the lab. The backend enforces honesty structurally. The UI is where
that survives or is quietly thrown away, so §5 is a requirements section, not a style guide.

## 2. Backend gap closure

Three gaps block the approved information architecture. All three are mechanical and mirror
patterns that already exist in the codebase. They land before any frontend work, as their own
commits, with tests.

### 2.1 `GET /api/v1/runs` and `GET /api/v1/collections`

Neither resource has a list endpoint today. A Collection is the product's deliverable — §10 of
the product spec ends its second success criterion at "exports a triaged Collection" — and
without a list, every Collection a user has ever made becomes unreachable the moment they close
the tab. Runs are nearly as bad: the API exposes `POST /runs/{id}/cancel`, which is only
meaningful if a Run outlives the page that started it.

Both follow `list_datasets.py` exactly: a `list` method on the port, a keyset-paginated
implementation on the SQLAlchemy repository, a use case that fetches `limit + 1` to decide
`next_cursor`, and a route returning `PaginatedResponse[T]`. `GET /runs` takes an optional
`kind` filter (`training` | `prediction`); the Runs screen shows predictions, while training
runs are reached through their Protocol.

### 2.2 A completed training Run must name its Protocol

`POST /protocols` returns a Run, and `RunResponse.protocol_id` is null for training runs
because the Protocol does not exist until training finishes. Nothing sets it afterwards. The
id is currently recoverable only by regex-ing it out of the blob path in `result_uri`, which
would couple the frontend to storage layout.

`Run` gains a nullable `protocol_id` column and a `link_protocol(protocol_id)` domain method
called at training completion. This respects the existing invariant that `params` is never
rewritten mid-flight — the repository's `update` deliberately does not persist `params`, and
this is a separate field. `RunResponse.protocol_id` then reads the column for both kinds,
replacing today's `params.get("protocol_id")` lookup, so prediction and training runs become
consistent.

Without this the training screen cannot navigate to the Scorecard it just produced, which is
the most important transition in the application.

### 2.3 Type the closed response shapes

Four fields are `dict[str, Any]` where the shape is closed, so the generated client would hand
the UI `unknown` precisely where the honesty layer lives:

| Field | Fix |
|---|---|
| `DatasetResponse.target` | reuse `TargetBody`, already defined in the same module |
| `DatasetResponse.split` | reuse `SplitBody`, likewise |
| `DatasetResponse.validation_report` | mirror `ValidationReport` / `InvalidRow` / `ConflictRow` as response models |
| `ProtocolResponse.readouts` | a `ReadoutResponse` model carrying `name`, `type`, `unit`, `direction`, `description` |

`ProtocolResponse.readouts` is the load-bearing one: it carries the `unit` and `direction` that
make a predicted IC50 comparable to a measured one. The handoff records that this pairing was
dropped at four separate boundaries during the backend build and fixed four times.

`ScorecardResponse.metrics`, `metrics_undefined`, `baseline_metrics` and
`ProtocolResponse.conditions` stay open maps. They genuinely are open maps — metric names vary
by task and conditions vary by engine.

The OpenAPI snapshot is regenerated afterwards and moves to `frontend/openapi.json`, which is
where `make generate-api` already writes and where every sibling keeps it.

## 3. Stack and conventions

`frontend/`, on port **3003** — the backend's `STUDIO_CORS_ORIGINS` already says so, and 3002 is
occupied by a sibling.

```
next ^16.2  react ^19.1  typescript ^5.8  tailwindcss ^4.1  @structflo/daikon-design-tokens ^1.0
@tanstack/react-query ^5.80  zustand ^5.0  orval ^7.10  radix-ui  react-hook-form + zod
ag-grid-community + ag-grid-react ^35.2   @rdkit/rdkit   papaparse   react-dropzone
biome  vitest + @testing-library/react
```

`src/app` (routes) + `src/features` (verticals) + `src/shared` (chrome, ui, lib, providers).
No top-level `src/components` or `src/lib` — the suite puts those under `shared/`. Tests are
colocated beside their source.

### chem-cellar is the template

`/Users/sidx/workspace/chem-vault2/frontend`. It is the chemistry sibling and the source of the
suite chrome standard that prot-cellar later received as a port. Copied rather than reinvented:
the chrome (sidebar, header, breadcrumbs, command palette, theme and font controls, settings
Appearance/About cards), the Sentinel auth chain, `custom-instance.ts` and `download.ts`, the AG
Grid wrapper and theme, the RDKit loader and structure renderers, the four providers, the zustand
stores, `toast.ts` / `utils.ts` / `query-defaults.ts`, `pnpm-workspace.yaml` (which exists solely
to carry pnpm 11's `allowBuilds` approvals — without it the Docker install fails), the
`predev` cache-capping script, and the vitest setup with its Node-25 localStorage polyfill.

Two details of the copy worth stating, because they are easy to get wrong: `globals.css` is two
`@import` lines plus app keyframes — every colour, radius and font comes from the tokens package,
and the app's only obligations are supplying the three font variables via `next/font` and driving
`data-theme` and `data-font` on `<html>`. And the provider nesting order
(Theme → FontFamily → Auth → Query) is load-bearing, because `AuthProvider` is what calls
`setApiBaseUrl()` before any query can fire.

shadcn's `form.tsx` is deliberately not among the vendored components; react-hook-form is used
directly with zod resolvers.

**Its convention on generated code is adopted deliberately:** orval generates *types*; hooks are
hand-written per feature against `customInstance`. Never hand-roll a type that mirrors a backend
DTO — alias the generated one. This matters here beyond consistency, because the triage grid's
datasource is an imperative `async (startRow) => …` callback that cannot be a React hook, and
because the 422 validation-report path needs handling the generated client does not know exists.

### Where we deliberately diverge

chem-cellar is a reference point, not a target. Five of its patterns are known problems and are
not reproduced:

1. **Polling.** Its async-job hooks run a `useEffect` + `setTimeout` chain whose dependency array
   holds an object derived from React Query data; when that derivation was not memoised it once
   produced 7000+ polls of a single job id. We use `refetchInterval: (q) => isTerminal(q.state.data?.status) ? false : 2000`.
   No effect, no dependency array, structurally immune.
2. **orval input.** Its orval reads a live `localhost:8000/openapi.json`, so codegen needs a
   running backend and CI cannot run offline. We read the committed snapshot.
3. **`middleware.ts`.** Its edge middleware is vestigial — its own comment explains that authz-mode
   tokens live in localStorage and cannot be validated at the edge, and that route protection is
   the client-side guard in the dashboard layout. It sets one header nothing reads, and the two
   env vars its docstring claims to use (`SENTINEL_URL`, `IDP_JWKS_URL`) are read by nothing at
   all. We do not create the file, and those two variables never enter `.env.example`.
4. **Env config.** Its `fetchAppConfig()` falls back to eleven `NEXT_PUBLIC_*` variables, of which
   five are declared, six are dead code, and several are misspelled relative to their `APP_*`
   counterparts (`APP_ENV` ↔ `NEXT_PUBLIC_ENVIRONMENT`, `APP_ENTRA_ID_TENANT_ID` ↔
   `NEXT_PUBLIC_ENTRA_TENANT_ID`). The branch only fires if a Next app fails to serve its own
   route. We drop it entirely: `/api/config` plus typed defaults, nothing else. Naming follows
   the namespaced `APP_SENTINEL_*` form rather than chem-cellar's flat `APP_GOOGLE_CLIENT_ID`,
   matching how the backend already namespaces `STUDIO_SENTINEL_*` and `STUDIO_IDP_*`.
5. **Lint debt.** Its biome config carries hundreds of warn-level burndown diagnostics, enough to
   hit biome's max-diagnostics cap and make `pnpm lint`'s exit code non-deterministic between runs
   on the same commit. Greenfield starts at zero: the burndown rules stay at error and the tree
   stays clean.

Not taken at all: Ketcher (nothing is drawn in Phase 1) and Plotly (see §5.4 — there is no
distribution to plot, so no charting library enters the project).

## 4. Information architecture

```
Dashboard
── Curate ──   Datasets · Protocols
── Apply ──    Runs · Collections
── Catalog ──  Engines
   (footer)    Settings
```

"Curate" because the product spec opens by calling this a curator's platform; the two groups are
the two things a user does — build a Protocol, or apply one somebody else published.

Creation flows are **linear wizards**, never a node graph. The product spec argues this
explicitly: a canvas lets a user wire an invalid graph, and the flow canvas belongs to Phase 2 as
a read-only lineage view.

### 4.1 Engines

Cards from `GET /engines`, one per engine, showing name, description, supported tasks and the
conditions it accepts. `is_baseline` is labelled. Read-only; engine registration is Phase 5.

### 4.2 Datasets

A list, and a four-step wizard: **file → columns → target → split**.

Column names come from parsing the CSV **in the browser** with papaparse (`preview: 5`), which
also yields a sample-rows preview for free. `POST /datasets/uploads` returns only an
`upload_ref`; there is no column-introspection endpoint and none is needed, because the file is
already in the browser.

The split step names both strategies with a plain-English line on what each simulates and
defaults to **scaffold**, the pessimistic one. Split is a scientific choice presented as such,
never a silent 80/10/10.

A 422 renders the **entire `ValidationReport`** as a full-page report — invalid rows with their
1-based positions in the original file, conflicting replicates with theirs, duplicates collapsed,
salts flagged, assay-noise spread — never collapsed to a toast. That report is the useful part of
a rejection.

Dataset detail shows the same report for an accepted Dataset, alongside content hash, row count
and split.

### 4.3 Protocols

A list distinguishing drafts from published by `status` and `is_locked`, a training form, and the
Scorecard.

The training form renders its condition inputs **entirely from the engine manifest** —
`key`, `label`, `type`, `required`, `default`, `minimum`, `maximum`, `options`, `help`. The labels
and help text are already written as biochemist-facing copy. Nothing about any engine is
hardcoded. The one exception is a two-entry table mapping engine tasks
(`regression` / `binary_classification`) to dataset target kinds (`numeric` / `binary`) so the
engine list can be filtered by the chosen Dataset; that mapping lives server-side only and is the
one piece of knowledge the self-describing catalogue failed to eliminate. It lives in one file.

Publishing is irreversible and its confirm dialog says so. A second publish returns 423.

### 4.4 Scorecard

The centrepiece. It reads top to bottom as an argument.

**Verdict band.** Three states, computed from the primary metric, the baseline metric and the
readout's `direction`:

- `baseline_is_self: true` → "This model is the baseline." No comparison is rendered, because
  none happened.
- beats baseline → "Beats the baseline", with both numbers and the delta.
- otherwise → "No better than the baseline", rendered prominently. Success criterion 3 is that
  this is immediately obvious on the first screen.

The band always names what the baseline is: ECFP4 + RandomForest on the same split.

**Three honesty cards**, equal height:

- *Optimism gap.* Three distinct states. `random_split_metrics` present → both numbers and the gap.
  Both `random_split_metrics` and `random_split_unavailable` null → not applicable, because the
  model was already trained on a random split. `random_split_unavailable` non-null → its message.
  "Not applicable" and "could not be computed" must read differently. `split_strategy` is
  authoritative; never infer it.
- *Assay-noise floor.* From within-duplicate spread, the honest floor for model error. Binary
  targets have no equivalent, so the card is simply absent rather than empty.
- *Applicability.* `applicability_coverage` as a percentage with a bar.

**Metric table.** Every metric, model beside baseline. For classification the primary metric leads
with MCC and balanced accuracy, never bare accuracy. **A null metric renders its
`metrics_undefined` reason inline** — those reasons are written for a scientist ("every row in the
test split has the same value — add positives, or split it differently"). A blank where a number
belongs is the exact failure this product exists to prevent.

**Where it fails.** `worst_rows` as rendered structures grouped by Murcko scaffold, each showing
actual, predicted and residual with units. This view is the point: a chemist should read "it fails
on the sulfonamides", not "R² = 0.61".

### 4.5 Runs

A list (predictions, newest first), a prediction wizard (published Protocol × CSV × optional
condition overrides), and a run page showing status, phase, progress and cancel.

Polling stops on `ready`, `failed` or `cancelled`. **A cache hit returns 202 with an
already-`ready` Run**, so the page branches on status and never assumes 202 means work started.

### 4.6 Triage

AG Grid over `GET /runs/{id}/results` using the infinite row model: one rendered structure per row,
each predicted readout with its unit, uncertainty, and an applicability flag. Sort, filter,
multi-select, then save the selection as a Collection.

`PredictionResponse` carries no id, but `POST /collections` wants `row_ids` as plain integer
offsets into the same paging order. The datasource stamps `__rowId = startRow + i` as each block
arrives and `getRowId` returns it, which is also what makes selection survive across blocks.

A scientist makes better decisions from "7 of your top 20 are outside the domain of applicability"
than from any aggregate metric, so that count is surfaced on the selection summary.

### 4.7 Collections

A list, a detail page (member count, provenance, the Run it came from), and CSV/SDF export.

Export is server-rendered (`GET /collections/{id}/export?format=csv|sdf`) and requires the auth
header, so it cannot be a plain `<a href>`. chem-cellar's `shared/lib/api/download.ts` already
solves this exactly — `downloadFile()` fetches with auth headers, reads the filename out of
`Content-Disposition`, and hands off to `saveBlob()`. It is copied as-is and also serves the
Download Template buttons through its `saveText()` sibling.

## 5. Cross-cutting requirements

### 5.1 Units and direction have one choke point

A single `<ReadoutValue value unit direction />` renders every number the API produces: Scorecard
metrics, worst rows, triage cells, collection previews. A predicted IC50 formats exactly like a
measured one because exactly one place decides how. This is the structural answer to a pairing
that was dropped four times during the backend build.

### 5.2 Absence renders as absence

XGBoost reports no uncertainty and it comes back `null`. Applicability is `null` when it cannot be
computed, never `0.0`. Both render as an em dash, never as zero and never as an empty cell. No
number is ever fabricated.

### 5.3 Never a bare blank

Anywhere a metric is null, its reason from `metrics_undefined` or
`random_split_metrics_undefined` renders in its place.

### 5.4 No charting library

The product spec asks for the applicability domain "as a distribution and as a per-compound flag".
The API returns a single `applicability_coverage` float on the Scorecard and a per-row
`applicability` on predictions — there is no distribution array to plot. It renders as a percentage
with a bar. This is a recorded gap against the product spec, not an oversight; a distribution would
need a new backend field.

## 6. House rules

Suite-wide UI rules that constrain these screens:

- **No UUIDs, as input or as display.** Dataset, Protocol and Engine references are named pickers.
  Runs are identified by Protocol name and date — a chemist says "the May 7th run". Breadcrumbs use
  an explicit `useBreadcrumbTrail`, never a URL-segment fallback that would print a UUID.
- **Every CSV import offers a Download Template** with correct headers and one or two realistic
  example rows, generated client-side as a Blob. Both imports here get one.
- **Explicit positive and negative buttons** on every create and confirm flow. Enter and Escape are
  accelerators, never the primary affordance. Save stays disabled until the form diverges from
  saved state.
- **No layout jump.** Wizard steps and dialog tabs keep equal pane min-heights and scroll
  internally; switching content inside a surface never resizes it.
- **No arrow glyphs** in link or button labels. Grid cards are equal height and aligned.
- **Count placement follows semantics.** Operate-on counts sit inside the button
  (`Save 47 as Collection`); forecast counts sit beside it (`312 rows will be rejected`).
- **No JSON textareas.** Engine conditions are real form controls driven by the manifest.

## 7. Data flow and error handling

`customInstance` is a plain fetch wrapper: it resolves the API base URL at runtime from
`/api/config`, merges Sentinel auth headers, serialises array params as repeated keys for FastAPI,
returns `undefined` on 204, and on non-2xx throws `ApiError` carrying `status` plus the **parsed
response body**. That retained body is what preserves the `ValidationReport` of a 422; its
human-readable `message` already flattens both FastAPI shapes (`detail: string` and Pydantic's
`detail: [{loc, msg}]`).

**The OpenAPI snapshot declares no error responses.** The 404, 409, 422, 423 and 503 the app really
returns are absent from the contract, so the generated types will not know about them and they are
handled by hand.

**`total_count` is always null.** Pagination is "load more", never "showing 1–50 of 320".

**Two cursor formats hide behind one field name.** Dataset, Protocol, Run and Collection lists use
base64 keyset cursors; `GET /runs/{id}/results` uses a plain integer offset. Both surface as
`next_cursor: string | null`. Cursors are treated as opaque and always passed through
`URLSearchParams`, which percent-encodes them — a `+` decoded as a space is the bug that caused an
infinite pagination loop in this codebase's sibling and is still live there.

**Virtualised and grid containers need a definite height** — the `flex flex-col h-full` → `flex-1
min-h-0` chain, or an explicit length. A percentage height through a stretched flex item is not
definite, and the grid silently renders every row.

Mutation errors surface through a single global `MutationCache.onError` toast; feature hooks add
only `onSuccess` invalidations and confirmations.

## 8. Testing

Colocated vitest on logic, not chrome:

- the Scorecard verdict computation, including `baseline_is_self` and direction handling
- the three optimism-gap states
- the triage row-id offset mapping
- CSV header parsing and template generation
- condition-form rendering from a manifest, covering each `ConditionResponse.type`
- the poll predicate stopping on each terminal status

Not snapshot tests of the sidebar. Backend additions in §2 get tests in the existing style.

Lint is verified by exit code on the files actually changed, with an explicit
`--max-diagnostics`, never by eyeballing piped output.

## 9. Branding

A geometric SVG `LogoMark` for Studio in the suite's visual language, "DAIKON Studio" set in
Overused Grotesk (the same `.woff2` the siblings ship), workspace slug beneath, on the sidebar
header and the login page.

Workspace memory uses the per-app localStorage namespace convention: `studio.lastWorkspaceId`.

## 10. Auth

Sentinel in authz mode, in the shared realm with chem-cellar, prot-cellar, daikon-gen3 and
docu-store. The service `daikon-studio-dev` is already registered and the backend's `.env` holds
its key; the frontend reuses the same key for its `/api/auth/mint` BFF route, exactly as
prot-cellar does. **No Sentinel changes are made by this work.** If role grants for `studio:read`,
`studio:write`, `studio:train` or `studio:publish` prove to be missing during QA, that is reported
rather than fixed.

## 11. Deferred

Recorded so their absence is a decision, not an omission:

- Lineage visualisation (Phase 2, needs the xyflow canvas)
- Sweep UI, additional splits, GPU engines (Phase 2)
- chem-cellar and prot-cellar as Sources, Proposal write-back, the global Catalog (Phase 3)
- Playwright E2E. chem-cellar has a mock-auth interception recipe worth adopting later; Phase 1
  verifies in a real browser against the running backend.
- An applicability-domain distribution (§5.4)
- Anything the product spec's own deferred list already assigns to a later phase

## 12. Success criteria

Inherited from the product spec, restated as UI outcomes:

1. A scientist uploads a CSV and publishes a working Protocol in under two minutes without reading
   documentation.
2. A second user in the same workspace runs that Protocol against their own CSV and exports a
   triaged Collection.
3. The Scorecard makes it immediately obvious when a Protocol is no better than the baseline.
4. Every noun in the product spec's §2 has been exercised end to end by a real user flow.

Criterion 4 is the goal; the others are how it is verified.
