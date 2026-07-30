# Handoff — daikon-studio, after the Phase 1 UI

**Written:** 2026-07-29, at the end of the frontend build.
**For:** a fresh session picking up the UX pass and whatever comes next.
**This is not a plan.** It is the state of the world plus the traps. Write the plan
yourself — brainstorm first, then `writing-plans`. The predecessor to this file,
`HANDOFF-frontend.md`, is still worth reading for the backend's own history.

---

## 1. Where things are

Everything is merged to **`main`** and pushed to
**`github.com/sidxz/daikon-studio`** (private). `feat/frontend` was
fast-forwarded into `main` and is now a dead branch — it exists on the remote
only as a marker and can be deleted whenever. Working tree clean.

The Phase 1 loop was exercised end to end in a browser against the live backend,
signed in as a real user in the **SACLAB-DEV** workspace:

upload a CSV → freeze a Dataset → train a Protocol against the mandatory baseline
→ read the Scorecard → publish → run it over new compounds → triage → save a
Collection → export CSV/SDF.

The Phase 2 UX pass then landed on top: server-side sort/filter/row-identity on
run results, native AG Grid filters and an in-domain switch on triage, the
Scorecard's verdict strip, an enriched predict flow, and theme-aware structure
thumbnails. **All of it has now been driven in a browser**, signed in as a real
user, including both paths §7 lists as previously unexercised — see §3 for what
that pass actually established.

| Area | State |
|---|---|
| Chrome, Sentinel auth, branding | done, live |
| Engines catalogue | done, live |
| Datasets: wizard, ValidationReport, list, detail | done, live |
| Protocols: training form, Scorecard verdict strip, publish | done, live — verdict strip seen for all three verdict kinds |
| Runs: predict wizard + preview, sort/filter/in-domain triage grid | done, live |
| Collections: list, detail, CSV/SDF export | done, live |

307 backend tests, 28 frontend unit tests, 2 Playwright E2E tests, `tsc` and
`biome` clean by exit code. `pnpm test` is vitest only — `tests/e2e/**` is
excluded from it, because Playwright specs are `.spec.ts` too and vitest will
otherwise collect them and fail on a foreign `beforeEach`. `pnpm test:e2e` runs
the browser suite.

Two documents matter more than this one:

| Document | Why |
|---|---|
| `specs/2026-07-29-daikon-studio-frontend-design.md` | The design, including every deliberate divergence from chem-cellar and why |
| `specs/2026-07-25-daikon-studio-phase-1-design.md` | The product. §2 is the terminology, and it is load-bearing |

---

## 2. Running it

```
make up        # Postgres 5435 + Valkey 6381 + migrations
make dev-be    # backend on 8002
make dev-fe    # frontend on 3003
```

`backend/.env` and `frontend/.env.local` both exist locally and are gitignored.
The frontend mirrors the backend's Sentinel service key; both use the
`daikon-studio-dev` service in the shared realm.

**The trap that cost the most time this session: a stale uvicorn serves a schema
that no longer matches the code you are reading.** A backend process started
before the day's changes had the new database column but not the new code, so
training runs completed with a null `protocol_id` and the new list endpoints
404'd — and every symptom pointed at the frontend. Before debugging any client
behaviour, check what the server actually exposes:

```
curl -s localhost:8002/openapi.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(sum(len(v) for v in d['paths'].values()), 'endpoints')"
```

21 is current. If you see 19, restart the backend.

---

## 3. What is NOT done

- ~~Browser verification of the Phase 2 UX pass~~ — **done.** The whole pass was
  driven live in SACLAB-DEV. What it established, in the order it matters:
  - **The `row_id` round trip holds under the hardest case.** With applicability
    filtered to ≥ 0.9 *and* log_solubility sorted descending, three rows tied at
    exactly −2.156 were selected and saved. The collection's snapshot Parquet
    holds precisely those three compounds. This is the behaviour the entire
    backend half of the pass exists to protect (see §4, traps 10 and 12).
  - **The stable-sort fix is visible.** Sorting the all-null Uncertainty column
    on an XGBoost run — where the whole frame is one tie group — returns exact
    original-file order, holds it across the 100-row block boundary, and
    produces no duplicate rows and no AG Grid console errors.
  - **Both previously unexercised paths are now exercised** (what was plan
    Task 6). A binary-classification Scorecard (BBBP-XGBoost) leads with MCC,
    says "No better than the baseline" at 0.603 against 0.632, and renders the
    noise-floor stat *absent* — the two remaining stats reflow, with no empty
    box. A `baseline_is_self` Scorecard (ESOL + ECFP4-RandomForest) says the
    model is the baseline, renders no comparison, and its metric table's third
    column reads "is the baseline" with no BASELINE header. Its RMSE 1.228 and
    MAE 0.924 are exactly the baseline figures the XGBoost Scorecard is measured
    against, so the two pages agree.
  - That last check earned its keep. The whole-branch review had found the
    rebuilt Scorecard rendering its honesty numbers in only one of three verdict
    branches, so a `baseline_is_self` protocol would have shown no optimism gap
    and no applicability coverage at all. The live page now shows all three,
    which is the fix confirmed rather than merely asserted.
  - Cache hits, the predict preview, the read-only trained-with conditions, the
    0.3 in-domain threshold on both sides, and structures in both themes all
    behave as designed. The console stayed clean throughout.
- **`Sweep`, lineage visualisation, cross-app Sources, Proposals, generation** —
  all Phase 2+ by the product spec, none started.
- ~~Playwright~~ — **the round trip is now a test, on both sides.**
  `backend/tests/api/test_triage_round_trip.py` reads a sorted and filtered page
  and asserts the saved Collection holds the structures that were on screen;
  `frontend/tests/e2e/triage-round-trip.spec.ts` asserts the browser posts the
  server's `row_id`s rather than page offsets. Both were validated by
  reintroducing the original defect and watching them go red — the backend one
  fails with `{'CCN'} == {'Fc1ccc(F)cc1'}`, which is the bug in the only terms
  that matter. Everything *else* in the bullets above is still hand-checked
  once and unencoded; the Scorecard's verdict branches are the obvious next
  spec, since that is the one a rework has already silently broken.
- **The E2E suite may never drive a real backend, and this is not negotiable.**
  It runs on port 3103 with `APP_API_BASE_URL` pointed at an unresolvable host,
  intercepts all of `/api/v1/**`, and carries unsigned payload-only JWTs that
  exist solely to satisfy the client's `exp` check. If a request 401s against a
  real server, **the 401 is correct** — mock the endpoint, or write the
  assertion as a pytest case where the auth harness is process-local. Do not add
  an env-gated JWKS override or a patched entrypoint to make it pass: the authz
  token carries `wid` and role claims, so that key would assert any identity on
  any workspace at any role, in a realm four other apps share. The reasoning is
  restated at the top of `frontend/tests/e2e/auth.ts` for whoever hits the 401s.
- **No CI.** Nothing runs the 307 backend, 28 vitest, or 2 Playwright tests on
  push — every suite above is only as good as someone remembering to run it.
- **A dashboard worth the name.** `/` is honest signposting, because nothing in
  the Phase 1 API aggregates anything. Real numbers need new endpoints.

---

## 4. Traps, all of them paid for

1. **AG Grid's infinite row model silently ignores `quickFilterText` and
   `autoHeight`**, and needs an explicit container height — a percentage through
   a flex parent is not definite, and the grid renders every row instead of
   windowing. It warns about all of this in the console; read those warnings.
2. **`Content-Disposition` is not exposed through CORS**, so JavaScript reading
   it cross-origin always gets null. Exports are named client-side now.
3. **The results cursor is a plain integer offset**, while every other list uses
   a base64 keyset cursor — behind the same `next_cursor` field name. A generic
   pagination helper cannot tell them apart. Since the Phase 2 pass it means
   "offset within the current filtered and sorted view," not offset into the
   whole file — the results file itself is immutable, but the view the offset
   counts into is not, which is why the client must reset to offset 0 on every
   filter or sort change rather than trying to translate an old offset forward.
4. **`total_count` is always null.** Pagination is load-more; there is no honest
   "1–50 of 320" to render.
5. **A cache hit returns 202 with an already-`ready` Run.** Branch on status;
   never assume 202 means work started.
6. **`valid_rows` on a ValidationReport counts rows whose structure parsed,
   before replicates are grouped.** It is not a compound count. Calling it one
   claimed 1,008 compounds for a dataset that trains on 997.
7. **`ConditionResponse.type` is a bare string, not an enum**, so codegen gives
   no union. The five values are in `features/engines/types`.
8. **Engines say `regression`/`binary_classification`; datasets say
   `numeric`/`binary`.** That two-entry translation lives in exactly one file,
   with a test that fails if the backend gains a third target kind.
9. **The OpenAPI snapshot declares no error responses.** 404/409/422/423/503 are
   all handled by hand.
10. **`row_id` comes from the server, minted before any filter or sort, and is
    the row's position in the *original* results file — not in whatever page
    or view the client is currently looking at.** The triage grid used to
    derive row identity from pagination offset (`startRow + index`); under a
    server-side sort or filter, that offset diverges from the true position
    silently, because reordering the view does not renumber the rows in it.
    Deriving `__rowId` that way again would break under exactly that
    condition, and the symptom would not be an error — it would be a
    Collection quietly holding the wrong compounds. This is the specific
    thing the browser has never confirmed; see §3.
11. **Both engines ignore `ctx.conditions` at predict time** — `predict()` on
    both `ecfp4_randomforest` and `ecfp4_xgboost` forwards to the same
    `_predict_with_tree_ensemble`, which never reads them. That is why the
    predict wizard shows a protocol's trained-with conditions read-only rather
    than as inputs — editable fields would be dead controls. If an engine ever
    starts reading predict-time conditions, this is the one place to check
    before making them editable.
12. **A sorted results page is only a stable window because the sort breaks ties
    on `row_id`.** Every page request re-reads the Parquet and re-sorts it, and
    polars' `sort` defaults to `multithreaded=True, maintain_order=False` — an
    unstable sort. Two page requests were therefore two independent
    permutations, so a tied row could come back in both pages or in neither.
    The worst case needs no hunting: `uncertainty` is null for every row of an
    XGBoost run, so sorting that column makes the whole frame one tie group.
    `result_view.py` now sorts on `[column, row_id]`, which is a total order and
    identical across requests by construction. There is a regression test, but
    it locks the invariant rather than proving the old bug — the installed
    polars happened to preserve input order in every trial, so the test does not
    fail against the pre-fix code. Do not cite it as evidence the old code was
    broken; the argument is the API contract, not the observation.
13. **"In domain" is 0.3, in all three places that say it.** The Scorecard's
    `applicability_coverage` is computed at `_APPLICABILITY_THRESHOLD = 0.3` in
    `build_scorecard.py`, and `predict_with_protocol.py`'s docstring promises
    that a compound the triage screen calls out-of-domain and one the Scorecard's
    coverage excludes are always the same compound. The frontend had drifted to
    `0.5` — in the applicability cell's warning colour and in the "N of your M
    are outside the domain of applicability" count — and the Phase 2 spec copied
    that 0.5 into the new "In domain only" switch, which would have made a
    Scorecard claiming 72% coverage sit beside a grid filtering at a different
    line. All three now read `IN_DOMAIN_FLOOR` from
    `features/runs/lib/result-query.ts`, which is 0.3. If that number ever moves,
    it moves on both sides of the stack at once or the promise above is a lie.

---

## 5. Sentinel: one unfixed bug, and a hard rule

**`POST /service-apps` is the only origin-affecting admin route in
identity-service that does not call `refresh_origins`.** Its five siblings all
do. So a service app created with `allowed_origins` set in one shot gets a
correct database row and a stale in-memory CORS allowlist: it looks
misconfigured while being configured perfectly. That is exactly what happened to
`daikon-studio-dev`, and it cost a diagnosis cycle before the cause was clear.

The workaround is Edit→Save on the service app in the admin UI, which fires
`PATCH` and does refresh. The fix is one line in `create_service_app` plus a test.

**It has not been applied.** The standing rule for this repo is no edits to
identity-service code, config, grants or deploys without explicit in-the-moment
consent — a multiple-choice pick is not consent. That realm serves four other
apps. Ask before touching it.

---

## 6. Conventions this codebase now has

Follow them or change them deliberately; do not drift.

- `src/app` routes are three-line re-exports. `src/features/<name>/` holds
  `components/`, `hooks/` (with a `query-keys.ts` declared once), `lib/` for pure
  tested logic, `types/`, and a hand-curated `index.ts` barrel — never `export *`.
- **orval generates types; hooks are hand-written** against `customInstance`.
  Alias a generated DTO under a domain name; never redeclare its shape.
- **Every number renders through `<ReadoutValue>`.** That is the single choke
  point keeping unit and direction attached, and it exists because that pairing
  was dropped four times during the backend build.
- **Absence renders as absence.** Null uncertainty and null applicability show an
  em dash, never zero.
- **No UUIDs, as input or display.** Named pickers everywhere; detail pages
  declare an explicit `useBreadcrumbTrail`; runs are identified by protocol and
  date. The triage grid's internal row offset is deliberately not a column.
- **Count placement follows semantics.** Operate-on counts inside the button
  (`Save 3 as collection`); forecast counts beside it (`1 of your 3 are outside
  the domain of applicability`).
- Both CSV imports offer a Download template, generated client-side.
- Verify lint and types **by exit code**, never by eyeballing piped output.

---

## 7. Test data, already staged

MoleculeNet benchmarks, shaped for upload, in this session's scratchpad and
`~/Downloads`:

| File | What | Use |
|---|---|---|
| `esol-solubility.csv` | 1008 rows, Delaney solubility | regression |
| `esol-holdout.csv` | 120 structures, no target | prediction runs |
| `bace-pic50.csv` | 1513 rows | regression |
| `bace-active.csv` | 1513 rows, binary | classification, and `baseline_is_self` if trained with ECFP4+RandomForest |
| `bbbp-permeable.csv` | 2050 rows, binary | exercises 11 unparseable structures and 105 salt flags |

Re-fetch from `https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/` if the
scratchpad is gone.

**Both of the once-unexercised paths have now been run** (§3): binary
classification via BBBP-XGBoost, and `baseline_is_self` via ESOL trained with
ECFP4+RandomForest. The workspace holds those protocols already, so re-checking
either after a Scorecard change costs nothing — and the `baseline_is_self` one is
worth re-checking after *any* change to the verdict branches, because that is the
branch a Scorecard rework has already silently broken once.

---

## 8. What the benchmark exposed, and why it matters

ESOL/XGBoost on a scaffold split: RMSE **1.200** against a baseline of **1.228**,
an optimism gap of **0.165**, and an assay noise floor of **0.151**. The model's
entire advantage over fingerprints-plus-random-forest is a fifth of the
measurement error, and less than a sixth of what an easier split would have
handed it.

The Scorecard had all three numbers on one screen and called it "Beats the
baseline" in green. That is precisely the flattery this product exists to
prevent, committed by the product itself. The verdict now has a `within-noise`
state that says the two models cannot be told apart with this data.

Worth keeping in mind while polishing: **every honest number on that page is one
the user did not ask for and would rather not see.** The design pressure will
always be toward making the Scorecard look better. It is supposed to look
uncomfortable when the model is not good.
