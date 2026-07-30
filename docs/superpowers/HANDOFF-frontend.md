# Handoff — daikon-studio frontend

**Written:** 2026-07-29, at the end of the Phase 1 backend build.
**For:** a fresh session building the studio UI.
**This is not a plan.** It is the state of the world plus the traps. Write the plan yourself —
brainstorm first, then `writing-plans`. The backend plan for this project is at
`docs/superpowers/plans/2026-07-25-phase-1-backend.md` and is a reasonable shape to imitate.

---

## 1. Start here

The backend is **complete and unmerged** on branch `phase-1-backend` (49 commits). 272 tests
passing, mypy strict clean, three import-linter contracts enforced with no exemptions.

**Decide first: merge to `main`, or build the frontend on the same branch?** Nobody has integrated
it yet. There is **no git remote configured** on this repo.

> **Superseded 2026-07-30.** Both questions are answered: the frontend was built on
> `feat/frontend`, that branch is merged to `main`, and the repo now pushes to
> `github.com/sidxz/daikon-studio` (private). The rest of this document is kept as the
> backend build's own history — read `HANDOFF-frontend-phase-2.md` for current state.

Read these two documents before anything else:

| Document | Why |
|---|---|
| `docs/superpowers/specs/2026-07-25-daikon-studio-phase-1-design.md` | The product. §2 is the terminology, which is load-bearing — Engine / Protocol / Run / Readout / Dataset / Collection all mean specific things, and the words were chosen to match how a biochemist already thinks. Use them in the UI copy. |
| `docs/superpowers/phase-1-deferred-items.md` | 44 consciously-deferred items. Several are frontend-facing; §6 below lists the ones that will bite you. |

---

## 2. What the UI has to do

The product's whole claim is **honesty**: it exists because no-code ML platforms produce models
that score beautifully on a random split and then fail in the lab. The backend enforces that
structurally. **The UI is where it either survives or is quietly thrown away**, so treat these as
requirements rather than nice-to-haves:

- **The baseline sits beside every result.** Every training run also trains ECFP4+RandomForest. If a
  scientist's model can't beat fingerprints-plus-random-forest, they must learn it on the first
  screen. `baseline_is_self: true` means the model *is* the baseline — render "this model is the
  baseline", never a comparison that didn't happen.
- **The optimism gap is visible.** On a scaffold split the backend also trains on a random split.
  Show both numbers. `random_split_metrics: null` + `random_split_unavailable: null` means "not
  applicable"; a non-null `random_split_unavailable` means "could not be computed". Those are
  different and must read differently. There is now a `split_strategy` field so you never have to
  infer this.
- **Never show a bare blank where a number belongs.** `metrics_undefined` and
  `random_split_metrics_undefined` are per-metric maps of *reasons*, written for a scientist
  (e.g. "every row in the test split has the same value — add positives, or split it differently").
  A null metric without its reason is the exact failure this product exists to prevent.
- **Units and direction, always attached.** Readouts carry `unit` and `direction` (`high`/`low`).
  A predicted IC50 must render the way a measured one does. This was dropped at four separate
  boundaries during the backend build and fixed four times — do not make it five.
- **Never fabricate a number.** XGBoost reports no uncertainty; it comes back `null`. Applicability
  is `null` when it can't be computed, never `0.0`. Render absence as absence.
- **The split is a scientific choice, not a setting.** Offer random and scaffold with a plain-English
  line on what each simulates. Default to the pessimistic one.
- **Worst-20 as structures.** The Scorecard's `worst_rows` carry structure and Murcko scaffold so a
  chemist reads "it fails on the sulfonamides" rather than "R² = 0.61". That view is the point.

The screens the design implies: engine catalogue, dataset upload wizard with its validation report,
a training form, the Scorecard, a triage grid, collection export. **Creation flows should be linear
wizards.** The spec argues this explicitly (§ the flow-UI discussion) — a node-graph editor lets a
user wire an invalid graph, and the flow canvas belongs to Phase 2 as a *read-only lineage view*.

---

## 3. The API

**19 endpoints.** `frontend-openapi.json` at the repo root is committed and verified byte-identical
to the running app. Generate the client from it — do not hand-write types.

```
GET    /api/v1/engines                              self-describing engine catalogue
POST   /api/v1/datasets/uploads                     multipart -> {upload_ref}
POST   /api/v1/datasets                             201 + ValidationReport
GET    /api/v1/datasets                             paginated
GET    /api/v1/datasets/{id}
POST   /api/v1/protocols                            202 -> a training Run
GET    /api/v1/protocols                            paginated
GET    /api/v1/protocols/{id}
GET    /api/v1/protocols/{id}/scorecard
POST   /api/v1/protocols/{id}/publish               204; second publish is 423
POST   /api/v1/runs                                 202 -> a prediction Run
GET    /api/v1/runs/{id}                            poll this
GET    /api/v1/runs/{id}/results                    paginated; the triage grid's data
POST   /api/v1/runs/{id}/cancel                     204
POST   /api/v1/collections                          201, from a triage selection
GET    /api/v1/collections/{id}
GET    /api/v1/collections/{id}/export?format=csv|sdf
GET    /health, /version                            unauthenticated
```

**`GET /api/v1/engines` is designed for you.** It returns everything needed to render both the
engine picker and each engine's condition form — per condition: `key`, `label`, `type`, `required`,
`default`, `minimum`, `maximum`, `options`, `help`. `label` and `help` are already written as
biochemist-facing copy. Render the form from this response; hardcode nothing.

**Long work is 202-then-poll.** Training and prediction return a Run immediately. Poll
`GET /runs/{id}` at 2s and stop on a terminal status (`ready` / `failed` / `cancelled`). There are
no WebSockets and none are planned. `progress` and `phase` are columns on the Run.

**Error bodies.** A shared handler maps domain errors to status codes: 404 not-found, 409 conflict,
422 validation, 423 locked (republish), 503 unavailable. A 422 from `POST /datasets` carries the
**entire `ValidationReport`** in `detail` — invalid rows with their 1-based original file positions,
conflicting replicates with theirs, duplicates collapsed, salts flagged, assay-noise spread. That
report *is* the useful part of a rejection; surface it, don't collapse it to a toast.

---

## 4. Stack and conventions

Every sibling frontend agrees; copy rather than invent. `prot-cellar/frontend` and
`chem-vault2/frontend` are the closest models — both are Next.js + orval + biome. `daikon-gen3/frontend`
uses eslint instead.

```
next ^16.2.2   react ^19.1.0   typescript ^5.8   tailwindcss ^4.1
@tanstack/react-query ^5.80   zustand ^5.0   orval ^7.10
ag-grid-react ^35.2  (the triage grid)      @rdkit/rdkit  (structure rendering)
@structflo/daikon-design-tokens ^1.0        — at /Users/sidx/workspace/daikon-design-tokens
```

- **orval against the committed snapshot**, not a live server. `prot-cellar/frontend/orval.config.ts`
  is the pattern; that project deliberately commits its `openapi.json` so diffs are reviewable and
  CI works offline. Ours is `frontend-openapi.json` at the repo root.
- **Design tokens** are a real package (`tokens.css`) — import it, don't re-pick colours.
- shadcn/radix components live locally (`components.json` in the siblings).

---

## 5. Environment

**Ports — verify before assuming, this bit has already bitten once:**

| | Port | Note |
|---|---|---|
| backend | 8002 | free |
| Postgres | 5435 | **not 5434** — that's `daikon-gen3-postgres-1` |
| Valkey | 6381 | free |
| **frontend** | **3003 or later** | **3002 is occupied.** The root `Makefile`'s `dev`/`dev-fe` still hardcode 3002 — fix that when you scaffold `frontend/`. |

Check with `lsof -nP -iTCP:<port> -sTCP:LISTEN`.

Setup: `cp backend/.env.example backend/.env`, then `make up` (Postgres + Valkey + migrations) and
`make dev-be`. `backend/README.md` documents the rest.

**Two Makefile traps, both documented but worth repeating:**
1. `make install` unconditionally runs `cd frontend && pnpm install` and **fails today** because
   `frontend/` doesn't exist. Scaffolding it fixes this; the README carries a caveat meanwhile.
2. **A JSON-list env var must be single-quoted in `.env`.** The Makefile sources with
   `set -a && . ./.env`, which strips quotes, so `STUDIO_CORS_ORIGINS=["http://localhost:3003"]`
   becomes unparseable and pydantic refuses to start the server. `.env.example` gets this right —
   don't "tidy" it. **Set CORS to your actual frontend port.**

**Auth is the real blocker, decide early.** Sentinel runs in the shared realm with chem-cellar,
prot-cellar, daikon-gen3 and docu-store. The backend has **no auth bypass, deliberately** — if the
service key is unset, the dependency reject-alls with 503 and the app fails at boot rather than
serving unauthenticated. Only `/health`, `/version`, `/docs`, `/openapi.json` are open. So you need
a reachable Sentinel, or an explicit dev-mode decision. The backend's own API tests mint real RS256
token pairs and stub only JWKS lookup — `backend/tests/api/conftest.py` shows how, and its comments
explain why it does not bypass the middleware. **Do not solve this by making the backend permissive.**

---

## 6. Frontend-facing gaps in the API

All deliberate, all recorded, none blocking — but you'll hit them.

1. **`total_count` is always `null`** in every `PaginatedResponse`. It is in the schema and assigned
   nowhere. Don't build "showing 1–50 of 320" against it.
2. **The OpenAPI snapshot declares no error responses** — only success codes plus FastAPI's 422. The
   404/409/423/503 the app really returns, and the `ValidationReport` body, are absent from the
   contract. Your generated client will not know about them; handle them by hand.
3. **Two cursor formats behind one field name.** Dataset and Protocol lists use base64 keyset
   cursors; `GET /runs/{id}/results` uses a plain integer offset. Both appear as
   `next_cursor: string | null`. A generic pagination helper cannot tell them apart. Treat cursors
   as opaque and always percent-encode them — a `+` decoded as a space is exactly the bug that
   caused an infinite pagination loop in this codebase and still exists in `prot-cellar`.
4. **Engine `tasks` and dataset `target.kind` use different vocabularies.** Engines advertise
   `regression` / `binary_classification`; a dataset's target says `numeric` / `binary`. The mapping
   lives only server-side. You need a two-entry translation table to filter engines by dataset —
   which is the one bit of hardcoded knowledge the self-describing catalogue failed to eliminate.
5. **`ConditionResponse.type` is a bare `string`**, not a closed enum, so codegen won't give you a
   union. Values are `string` / `integer` / `number` / `enum` / `bool`.
6. **A cache hit returns 202, not 200**, with an already-`ready` Run. Don't assume 202 means "work
   started".
7. **Draft vs published.** `GET /protocols` returns drafts too — a draft is not runnable by anyone
   else. Distinguish them by `status` and `is_locked`. Publishing is irreversible.

---

## 7. Test-suite state, if you touch the backend

254 → 272 tests, all green, but the *test-idiom* layer is the most-drifted part of the codebase and
the final review flagged it: three near-duplicate dataset→train→publish ladders, **two different
classes both named `Studio`**, six copies of a savepoint-rollback recipe, two in-memory blob-store
fakes neither of which lives in `tests/fakes/`, and no `tests/factories/` despite the spec naming it.
If you add backend tests, one consolidation pass into `tests/api/conftest.py` and `tests/fakes/`
would pay for itself before the suite doubles again.

Also: `backend/tests/api/conftest.py` builds a fresh DI container per test. **Do not hoist it to
session scope** as a speed optimisation — it currently hides a class of cross-wiring bug, and the
comment there says so.

---

## 8. State of the machine as I leave it

- A dev server may still be running: uvicorn on 8002, Postgres 5435, Valkey 6381. `make down` stops
  the containers. `backend/.env` exists locally and is gitignored.
- 42 per-task reports are in the gitignored `.superpowers/sdd/2026-07-25-phase-1-backend/`
  directory — every finding, fix and ruling from the backend build, if you ever need to know why
  something is the way it is.
- ~~Nothing is merged. Nothing is pushed. There is no remote.~~ All three were true when
  this was written; none is true now — see the note at the top.
