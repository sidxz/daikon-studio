# Handoff — daikon-studio, after the Phase 1 UI

**Written:** 2026-07-29, at the end of the frontend build.
**For:** a fresh session picking up the UX pass and whatever comes next.
**This is not a plan.** It is the state of the world plus the traps. Write the plan
yourself — brainstorm first, then `writing-plans`. The predecessor to this file,
`HANDOFF-frontend.md`, is still worth reading for the backend's own history.

---

## 1. Where things are

Everything is on **`feat/frontend`**, 21 commits ahead of `main`, working tree clean.
**Nothing is pushed — there is still no git remote.** `main` holds the Phase 1
backend, which was merged there at the start of the prior session.

The Phase 1 loop was exercised end to end in a browser against the live backend,
signed in as a real user in the **SACLAB-DEV** workspace:

upload a CSV → freeze a Dataset → train a Protocol against the mandatory baseline
→ read the Scorecard → publish → run it over new compounds → triage → save a
Collection → export CSV/SDF.

The Phase 2 UX pass then landed on top: server-side sort/filter/row-identity on
run results, native AG Grid filters and an in-domain switch on triage, the
Scorecard's verdict strip, an enriched predict flow, and theme-aware structure
thumbnails. **None of it has been driven in a browser** — see §3 for why and
exactly what that leaves unverified.

| Area | State |
|---|---|
| Chrome, Sentinel auth, branding | done, live |
| Engines catalogue | done, live |
| Datasets: wizard, ValidationReport, list, detail | done, live |
| Protocols: training form, Scorecard verdict strip, publish | done, not seen live since the verdict-strip rework |
| Runs: predict wizard + preview, sort/filter/in-domain triage grid | done, not seen live since the UX pass |
| Collections: list, detail, CSV/SDF export | done, live |

303 backend tests, 28 frontend tests, `tsc` and `biome` clean by exit code.

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

- **Nothing in the Phase 2 UX pass has been verified in a browser.** The
  controller could not sign in — auth is Google OAuth, and there is no way to
  do that on the user's behalf — so every task's browser-verification step was
  skipped. The pass shipped on unit/API tests, `tsc`, and `biome` alone. Two
  things specifically need eyes before they're trusted: the triage grid's
  filter and sort behaviour under AG Grid's infinite row model, and — the one
  that matters most — the save-as-collection round-trip while a filter or sort
  is active. That round-trip is the exact behaviour `row_id` was added to
  protect (see §4), and it has never been exercised end to end.
- **Task 6 of the UX-pass plan did not run.** That was the live pass over the
  two paths no browser has ever seen: a binary-classification protocol end to
  end, where the noise-floor stat must be *absent* from the verdict strip, not
  rendered empty, and `baseline_is_self`, where the strip must say the model
  IS the baseline and render no comparison at all. Both still have only
  unit-test coverage. The files are staged and ready — `bace-active.csv` for
  the first, the same file trained with ECFP4+RandomForest for the second (see
  §7) — this is owed to the user, not dropped. It is also worth more than it
  looks: the whole-branch review found that the rebuilt Scorecard rendered its
  honesty numbers in only one of three verdict branches, so a `baseline_is_self`
  protocol showed no optimism gap and no applicability coverage anywhere on the
  page. That is precisely what Task 6 step 3 would have caught. It is fixed, and
  the fix is itself unverified in a browser.
- **`Sweep`, lineage visualisation, cross-app Sources, Proposals, generation** —
  all Phase 2+ by the product spec, none started.
- **Playwright.** No E2E. chem-cellar has a mock-auth interception recipe at
  `chem-vault2/.claude/skills/verify/SKILL.md` worth adopting — it would also
  be the fix for the Google-OAuth sign-in wall above.
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

**Two paths are still unexercised and worth hitting early:** a binary
classification protocol end to end (MCC and balanced accuracy lead there, and the
noise-floor card should be *absent*, not empty), and `baseline_is_self`, where the
Scorecard must say "this model is the baseline" rather than render a comparison
that never happened. Both have unit tests; neither has been seen in a browser.

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
