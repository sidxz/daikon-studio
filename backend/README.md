# daikon-studio backend

An in-silico protocol studio: freeze a validated, split Dataset from a CSV of
structures, train a Protocol with an honest Scorecard (mandatory baseline,
visible optimism gap on a scaffold split), publish it, run it against new
compounds, triage the results, and save/export a Collection. See
`.superpowers/sdd/2026-07-25-phase-1-backend/` for the full design.

All commands below are `make` targets, run from the **repo root** (the single
`Makefile` there wraps `cd backend && uv run ...` internally) unless noted
otherwise. `make help` lists every target with a one-line description.

## First-time setup

```bash
cp backend/.env.example backend/.env   # see the note on STUDIO_CORS_ORIGINS below
make install                           # backend deps; frontend half fails today, see below
make up                                # start Postgres, run migrations, seed the dev runners
make dev-be                            # backend only, on :8002
```

`backend/.env` does not exist in a fresh clone and nothing creates it for
you -- it's gitignored (per-developer, and it holds a real secret in any
non-local environment) -- so `make up`, `make dev`/`make dev-be`, and
`make migrate` will fail or silently fall back to mismatched defaults until
you copy `.env.example` yourself. Do that first.

**No `frontend/` directory exists yet** (the frontend plan hasn't been
written -- see "OpenAPI snapshot" below for why). `make install` runs the
backend's `uv sync` first, then unconditionally `cd frontend && pnpm
install`; today that second line fails outright:

```
cd backend && uv sync
Resolved 98 packages in 11ms
cd frontend && pnpm install
/bin/sh: line 0: cd: frontend: No such file or directory
make: *** [install] Error 1
```

That's expected and harmless for backend-only work: the backend half already
completed by the time the frontend half fails, `uv sync` is idempotent, and
nothing downstream in this README needs the frontend. `make dev` has the
same asymmetry for the same reason -- its frontend leg is backgrounded, so it
fails silently into `.logs/frontend.log` instead of aborting `make dev`
itself, but the backend and worker still start fine. Use `make dev-be` +
`make dev-worker` for backend-only work instead of `make dev` until
`frontend/` exists. Both caveats go away once the frontend is scaffolded.

**`STUDIO_CORS_ORIGINS` must stay single-quoted in `.env`.** It's a JSON
list (`'["http://localhost:3003"]'`), and the Makefile loads `.env` into the
recipe shell with `set -a && . ./.env && set +a`. Without the single quotes,
plain shell word-splitting mangles the value before pydantic-settings ever
sees it, and the backend refuses to start with `SettingsError: error parsing
value for field "cors_origins"`. `.env.example` already has this right --
just don't strip the quotes when you edit it.

## Ports

| Service    | Port | Notes                                            |
|------------|------|---------------------------------------------------|
| Backend    | 8002 | `uvicorn`, `--reload`; http://localhost:8002/docs |
| Postgres   | 5435 | not 5432/5434 -- those are taken by sibling projects on this machine |
| Frontend   | 3003+ | 3002 is occupied; the frontend (once scaffolded) picks 3003 or later |

## Day to day

```bash
make dev            # backend + frontend + both runner agents, backgrounded
make dev-be          # (re)start just the backend
make dev-worker       # (re)start just the default-lane runner agent
make logs            # tail all logs
make stop             # stop the backgrounded dev processes
make migrate          # apply alembic migrations
```

## Running without a runner agent

Training and prediction runs normally go through a self-hosted runner agent
that claims work over HTTP (`make dev-worker`/`make dev-worker-gpu`). Set
`STUDIO_INLINE_JOBS=1` (already the default in `.env.example`) to run those
jobs in-process instead, synchronously, inside the same request/test that
submitted them -- no runner needed at all. This is what the test suite uses
(`tests/api/conftest.py`'s `app` fixture always sets `inline_jobs=True`);
it's also the fastest way to run the backend locally with no agent process
running. Unset it (or set it to `0`) to exercise the real runner-backed path.

## Tests

```bash
make test       # unit tests + import-linter (fast, no containers)
make test-api   # API tests (real Postgres via testcontainers, real auth middleware)
make test-all   # the whole suite (unit + api + integration) + import-linter
```

`tests/integration/test_full_loop.py` is the acceptance test: one continuous
HTTP journey -- upload, Dataset, Protocol, Scorecard, publish, prediction
Run, triage results, Collection, export -- through the real app, real auth,
and a real (ephemeral, testcontainers) Postgres. It's the test that decides
whether "Phase 1 backend is done."

`make lint` (ruff + mypy) and `uv run lint-imports` (the Clean Architecture
and bounded-context-independence contracts) must both pass; run `make lint`
from the repo root.

## OpenAPI snapshot

`frontend-openapi.json` at the repo root is a committed, generated snapshot
of this backend's OpenAPI schema -- the input the (not-yet-written) frontend
plan's orval config generates a TypeScript client from. Regenerate it after
any route/schema change:

```bash
cd backend && uv run python -c "
import json
from daikonstudio.interface.app import create_app
print(json.dumps(create_app().openapi(), indent=2))
" > ../frontend-openapi.json
```

It's committed (not gitignored) so a diff on it is reviewable in a PR and CI
can generate the frontend client offline, without a live backend.
