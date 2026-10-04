# daikon-studio

An in-silico protocol studio for drug discovery: freeze a validated, split
Dataset from a CSV of structures, train a Protocol with an honest Scorecard
(mandatory baseline on the identical split, optimism gap, assay-noise floor,
bootstrap interval under the headline number), publish it, run it against new
compounds on a self-hosted runner, triage the results, and save or export a
Collection. The backend is FastAPI + SQLAlchemy + polars; the frontend is
Next.js. Design notes live in `docs/` (`roadmap.md`, `engine-research.md`,
`study-replication.md`) and the beta-readiness plan in `docs/plans/`.

Every command below is a `make` target run from the **repo root**; `make help`
lists them all.

## First-time setup

```bash
cp backend/.env.example backend/.env     # fill in STUDIO_DUAR_SERVICE_KEY and STUDIO_IDP_AUDIENCE
cp frontend/.env.example frontend/.env.local   # same key as APP_DUAR_SERVICE_KEY, same client id
make install                             # backend (uv, with the gpu and s3 extras) + frontend (pnpm)
make up                                  # Postgres on :5437, migrations, the two dev runner rows
make dev                                 # backend :8002, frontend :3003, default + gpu lane runners
open http://localhost:3003
```

`backend/.env` is gitignored and nothing creates it for you. `STUDIO_CORS_ORIGINS`
must stay single-quoted there: the Makefile sources the file into a shell, and an
unquoted JSON list is word-split before pydantic-settings sees it.

Auth is the shared Duar identity service (realm `daikon-siblings`); it is the one
dependency that does not run on your laptop, and the API refuses to boot without a
service key, by design. Sign-in is a real Google account.

**After any `uv add`, run `make install` again.** A bare `uv sync` drops the gpu
and s3 extras; the symptom is `make lint` failing on mypy and both local runners
advertising engines they cannot import.

## Ports

| Service  | Port  | Notes                                                        |
|----------|-------|--------------------------------------------------------------|
| Backend  | 8002  | `uvicorn --reload`; http://localhost:8002/docs               |
| Frontend | 3003  | `next dev`                                                   |
| Postgres | 5437  | 5432/5434/5435/5436 belong to sibling projects on this machine |

## Day to day

```bash
make dev            # everything, backgrounded; logs in .logs/
make dev-be         # restart just the backend
make dev-fe         # restart just the frontend
make dev-worker     # restart the default-lane runner agent (it does NOT hot-reload)
make dev-worker-gpu # restart the gpu-lane runner agent (chemprop, MoLFormer; MPS on a Mac)
make logs           # tail all four logs
make stop           # stop the dev processes
make migrate        # alembic upgrade head
```

The runner agents do not hot-reload: after any engine or training change, restart
them or you will debug a fix that never loaded. Set `STUDIO_INLINE_JOBS=1` in
`backend/.env` to run jobs inside the API request instead (what the tests do); with
it set, the runner agents are idle and a chemprop fit blocks the browser.

## Tests and gates

```bash
make test       # backend unit tests + import-linter
make test-api   # API tests (real Postgres via testcontainers, real auth middleware)
make test-all   # unit + api + integration + import-linter
make lint       # ruff + ruff format --check + mypy strict
make test-fe    # vitest
make lint-fe    # biome
make generate-api   # regenerate frontend/openapi.json and the orval types after any route or schema change
```

`tests/integration/test_full_loop.py` is the acceptance test: one HTTP journey from
upload to export through the real app, real auth and a real Postgres. CI
(`.github/workflows/ci.yml`) runs every gate above, builds both images, and
boot-checks them.

With the gpu extra installed, the full unit suite can segfault on macOS (three
OpenMP runtimes in one process; `docs/roadmap.md`, Traps). `OMP_NUM_THREADS=1` is
already set by the Makefile; if it still dies, run the engine test files one process
at a time.

## Images and deployment

```bash
make image-smoke      # build daikon-runner:cpu (API + default-lane runner) and import the app and every engine in it
make image-frontend   # build daikon-frontend:local
```

A green `docker build` says nothing about whether an image works: the CPU image
built clean for two months while LightGBM could not import inside it. `image-smoke`
is the check that would have caught it; CI runs the same check on every push.

`deploy/README.md` is the runbook for a single-host production stack (Caddy, API,
migrate step, frontend, Postgres, a default-lane runner) via `deploy/compose.yml`.

The GPU runner image (`backend/Dockerfile.gpu`, CUDA, x86_64 only) is built on
atlantic, which has the NVIDIA GPU, over the `atlantic` docker context: `make
image-runner-gpu`. `make publish-runner-gpu` also runs it on the GPU, Trivy-scans it
and pushes it to ghcr, and is part of a backend release (RELEASING.md).

## Layout

```
backend/src/daikonstudio/
  domain/         pure aggregates (catalog, data, execution, runners); no framework imports
  application/    use cases, ports, the engine contract and registry, training and prediction handlers
  infrastructure/ SQLAlchemy, fsspec blobs, the engines, the runner agent, Duar, logging
  interface/      FastAPI routes, middleware, error handlers
frontend/src/
  app/            Next.js routes
  features/       one vertical per noun: datasets, protocols, runs, collections, sweeps, runners, engines
  shared/         chrome, ui, api client, auth, providers
```

The import-linter contracts (`backend/pyproject.toml`) enforce the layer order and
the independence of the three bounded contexts; `make lint` runs them.
