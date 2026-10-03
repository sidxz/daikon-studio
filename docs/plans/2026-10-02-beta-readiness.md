# Beta Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every finding of the 2026-10-02 beta-readiness audit so a single-lab, single-workspace beta of daikon-studio can be deployed from `main` with one `docker compose up`.

**Architecture:** The application is complete; this plan adds the layer around it. Deployable images that boot (API, default runner, frontend), a host-agnostic Compose stack behind Caddy, CI that builds and smoke-runs those images, structured logging with request ids and a real readiness probe; then the day-one defects in order of how soon a beta user hits them (one-hour session expiry, stuck screens, compound identifiers lost at prediction, dataset inputs accepted and failed later, missing retry, unguarded deadlines, a naked "beats the baseline" claim), and finally the docs.

**Tech Stack:** Python 3.13 / FastAPI / SQLAlchemy async / polars / structlog; Next.js 16 / React 19 / TanStack Query 5 / `@duar-auth` 1.0; Docker, Compose, Caddy 2; GitHub Actions.

**Spec:** `docs/plans/2026-10-02-beta-readiness-audit.md` (the audit findings this plan closes). Companion design docs live in git history under `50860b2^:docs/superpowers/specs/` (retry: `2026-08-04-retry-a-failed-run-design.md`; runners: `2026-08-04-self-hosted-runners-design.md`).

## Global Constraints

- Work on branch `beta-readiness`, branched from `main`. Never commit to `main` directly. Never push without being asked.
- Commit messages: conventional (`feat(scope): …`, `fix(scope): …`, `chore: …`, `docs: …`), ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. **Never** add a `Claude-Session:` trailer.
- Backend: ruff line-length 99, mypy `strict`, import-linter contracts (`interface > infrastructure > application > domain`; domain imports no framework). `make lint` and `uv run lint-imports` must pass after every backend task.
- Every request body model is `extra="forbid"`. `workspace_id` is never read from a body or URL.
- Any route or schema change ends with `make generate-api` (regenerates `frontend/openapi.json` and the orval types) in the same commit.
- Frontend: `pnpm lint` (biome), `pnpm exec tsc --noEmit`, `pnpm test` must pass after every frontend task. House rules: no UUIDs displayed, absence renders as an em dash never zero, counts inside buttons are operate-on counts, no JSON textareas.
- Tests: backend `env OMP_NUM_THREADS=1 uv run pytest tests/unit`, `uv run pytest tests/api tests/integration` (testcontainers Postgres, Docker must be running). Run the file you touched first, the suite before committing.
- Do **not** build on the remote `ned`/`orca` machines. The GPU image rebuild is handed back to the human as a command.
- Ponytail is active: the minimal change that works, in the file that already owns the behaviour. Mark deliberate ceilings with `# ponytail:` comments.
- Trust-model decision (recorded, not re-litigated): runners stay instance-level, "one lane = one trust domain"; token minting and revocation move to the admin role. Deployment target decision: host-agnostic Compose + Caddy.

## Review Focus

Inputs the audit implies but no existing test exercises, most likely to bite first. Each has its pinning test in the task named.

1. **A CSV whose header carries a UTF-8 BOM** (`﻿smiles`). Expected: the column is found and the file is accepted. Task 10, `test_a_bom_on_the_first_header_is_stripped`.
2. **A numeric target containing `NA`, `<10` or `12,5`.** Expected: those rows are rejected with their row numbers and values in the ValidationReport; the file is otherwise accepted. Never a 500. Task 10, `test_non_numeric_target_values_are_invalid_rows_not_a_crash`.
3. **A 401 arriving on an API call an hour into a session.** Expected: exactly one silent re-auth starts and the user lands back where they were; no error toast. Task 16, `custom-instance.test.ts`.
4. **A prediction upload whose chosen identifier column has blanks and duplicates.** Expected: identifiers are carried verbatim as text, a blank becomes null, duplicates both survive; nothing is used as a key. Task 11, `test_a_blank_identifier_is_null_and_duplicates_both_survive`.
5. **A tree-engine training run that overruns its deadline between fits.** Expected: the run ends FAILED with the deadline reason, not left RUNNING. Task 12, `test_progress_between_fits_honours_the_deadline`.

---

### Task 1: Branch, and land the inherited working tree

**Files:**
- Rename: `backend/tests/unit/infrastructure/test_sentinel_scope.py` → `backend/tests/unit/infrastructure/test_duar_scope.py`
- Already modified (commit as-is): `Makefile`, `backend/.env.example`, `backend/README.md`, `backend/src/daikonstudio/infrastructure/duar/auth.py`, `backend/src/daikonstudio/interface/app.py`, `backend/src/daikonstudio/settings.py`, `docker-compose.yml`

- [ ] **Step 1: Branch**

```bash
cd /Users/sidx/workspace/daikon-studio && git checkout -b beta-readiness
```

- [ ] **Step 2: Rename and format the untracked test**

```bash
mv backend/tests/unit/infrastructure/test_sentinel_scope.py backend/tests/unit/infrastructure/test_duar_scope.py
cd backend && uv run ruff format tests/unit/infrastructure/test_duar_scope.py && uv run ruff check tests/unit/infrastructure/test_duar_scope.py
```

- [ ] **Step 3: Run the test**

Run: `cd backend && uv run pytest tests/unit/infrastructure/test_duar_scope.py -q`
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "fix(dev): move Postgres to 5437; log the effective Duar scope at boot

The scope warning makes the silent auth wedge (whoami failing at startup,
every request 403ing with a clean log) visible. Test renamed for Duar.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Lint gates that pass in any venv, and autogenerate that sees every table

**Files:**
- Modify: `backend/mypy.ini`
- Modify: `backend/alembic/env.py:14-21`
- Modify: `frontend/src/shared/lib/auth/config.ts` (biome autofix only)
- Test: `backend/tests/integration/test_migrations.py`

**Interfaces:**
- Produces: nothing new; `make lint` and `pnpm lint` exit 0 with or without the gpu extra installed.

- [ ] **Step 1: Write the failing migration-parity test**

Append to `backend/tests/integration/test_migrations.py` (reuse that file's migrated-engine fixture name; it is the fixture the existing test in the file already takes):

```python
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from daikonstudio.infrastructure.persistence.sqlalchemy.base import Base
from daikonstudio.infrastructure.persistence.sqlalchemy.catalog import models as _catalog  # noqa: F401
from daikonstudio.infrastructure.persistence.sqlalchemy.data import models as _data  # noqa: F401
from daikonstudio.infrastructure.persistence.sqlalchemy.execution import models as _execution  # noqa: F401
from daikonstudio.infrastructure.persistence.sqlalchemy.runners import models as _runners  # noqa: F401


async def test_no_table_exists_only_in_the_orm_or_only_in_the_migrations(migrated_engine):
    """`alembic revision --autogenerate` emits `drop_table` for any table the
    migrations created that `Base.metadata` does not know about. Pins that
    every model module is imported where autogenerate can see it."""

    def diff(connection):
        return compare_metadata(MigrationContext.configure(connection), Base.metadata)

    async with migrated_engine.connect() as connection:
        diffs = await connection.run_sync(diff)
    table_diffs = [d for d in diffs if d[0] in ("add_table", "remove_table")]
    assert table_diffs == [], table_diffs
```

- [ ] **Step 2: Run it**

Run: `cd backend && uv run pytest tests/integration/test_migrations.py -q`
Expected: PASS (the test imports the runners model itself, so it passes; what it pins is that nobody removes the import from `env.py` and then trusts autogenerate). Now make `env.py` match.

- [ ] **Step 3: Import the runners model in `alembic/env.py`**

After the `execution` import at line 19-21 add:

```python
from daikonstudio.infrastructure.persistence.sqlalchemy.runners import (
    models as runners_models,  # noqa: F401
)
```

- [ ] **Step 4: Make mypy independent of the gpu extra**

Append to `backend/mypy.ini`:

```ini
[mypy-torch.*]
ignore_missing_imports = true

[mypy-lightning.*]
ignore_missing_imports = true

[mypy-transformers.*]
ignore_missing_imports = true

# The three gpu-lane modules subclass classes from optional packages. With the
# extra installed they type-check for real; without it those bases are Any, and
# strict mode would reject the subclass. Either venv must lint green.
[mypy-daikonstudio.infrastructure.engines._lightning]
disallow_subclassing_any = false

[mypy-daikonstudio.infrastructure.engines.molformer_xl]
disallow_subclassing_any = false
warn_unused_ignores = false

[mypy-daikonstudio.infrastructure.engines.chemprop_dmpnn]
disallow_subclassing_any = false
warn_unused_ignores = false
```

- [ ] **Step 5: Fix the two biome errors**

Run: `cd frontend && pnpm lint:fix && pnpm lint`
Expected: `Checked 131 files … No fixes applied. Found 0 errors` on the second command.

- [ ] **Step 6: Verify both gates**

Run: `make lint && cd backend && uv run lint-imports`
Expected: ruff clean, `Success: no issues found`, 3 contracts kept.

- [ ] **Step 7: Restore the dev venv's extras (environment, not a commit)**

Run: `make install` (background is fine; it only affects the local venv). Then `make dev-worker && make dev-worker-gpu` so the running agents can import what they advertise.

- [ ] **Step 8: Commit**

```bash
git add backend/mypy.ini backend/alembic/env.py backend/tests/integration/test_migrations.py frontend/src/shared/lib/auth/config.ts
git commit -m "chore: lint gates independent of the gpu extra; autogenerate sees the runners table

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: An API image that boots

**Files:**
- Modify: `backend/Dockerfile`
- Modify: `backend/Dockerfile.gpu` (the apt line)
- Modify: `Makefile` (new `image-smoke` target)

**Interfaces:**
- Produces: image `daikon-runner:cpu` whose `python -c "import daikonstudio.interface.app"` succeeds; `make image-smoke`.

- [ ] **Step 1: Prove the failure**

Run: `make image-runner-cpu && docker run --rm daikon-runner:cpu python -c "import daikonstudio.infrastructure.engines.registry"`
Expected: `OSError: libgomp.so.1: cannot open shared object file`

- [ ] **Step 2: Install the OpenMP runtime and prepare the blob mount point**

In `backend/Dockerfile`, directly after `WORKDIR /app`:

```dockerfile
# libgomp1 is load-bearing. LightGBM's wheel links the system OpenMP runtime
# (libgomp.so.1) and python:3.13-slim does not ship it, so without this line the
# image builds green and the API dies at import. Found 2026-10-02 by booting the
# image, not by building it -- the same trap docs/roadmap.md records for the GPU
# image. `make image-smoke` now runs that boot check for you.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
```

Replace the `RUN useradd … && chown -R studio /app` line with:

```dockerfile
# /data/blobs is the default STUDIO_BLOB_BASE_URL. Created and owned here so a
# named volume mounted on it inherits writable ownership for uid 10001.
RUN useradd --create-home --uid 10001 studio \
 && mkdir -p /data/blobs \
 && chown -R studio /app /data
```

In `backend/Dockerfile.gpu`, change the apt install line to:

```dockerfile
 && apt-get install -y --no-install-recommends libxrender1 libxext6 libexpat1 libgomp1 \
```

- [ ] **Step 3: Add the smoke target to the Makefile**

After `image-runner-gpu`:

```makefile
image-smoke: image-runner-cpu ## Build the CPU image and prove it can import every engine and the app
	docker run --rm -e STUDIO_DUAR_SERVICE_KEY=smoke -e STUDIO_IDP_AUDIENCE=smoke daikon-runner:cpu \
		python -c "import daikonstudio.interface.app, daikonstudio.infrastructure.engines.registry as r; print('engines:', sorted(m.id for m in r.default_registry().manifests()))"
```

Add `image-smoke` to the `.PHONY` list.

- [ ] **Step 4: Run it**

Run: `make image-smoke`
Expected: last line `engines: ['chemprop-dmpnn', 'descriptors-xgboost', 'ecfp4-lightgbm', 'ecfp4-randomforest', 'ecfp4-xgboost', 'molformer-xl', 'tanimoto-gp']`

- [ ] **Step 5: Commit**

```bash
git add backend/Dockerfile backend/Dockerfile.gpu Makefile
git commit -m "fix(docker): install libgomp1 so LightGBM imports; make image-smoke boots the image

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: A frontend image

**Files:**
- Create: `frontend/Dockerfile`
- Modify: `frontend/.dockerignore` (remove the `Dockerfile` line so the build context stays valid; keep the rest)
- Modify: `Makefile` (new `image-frontend` target)

**Interfaces:**
- Produces: image `daikon-frontend:local` serving on `PORT` (default 3000) with `/api/config`, `/login`, `/RDKit_minimal.wasm`.

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
# syntax=docker/dockerfile:1
# Next.js standalone build. Runtime config is read from APP_* env at request
# time (src/app/api/config/route.ts), so one image serves every environment.

FROM node:22-alpine AS build
RUN corepack enable && corepack prepare pnpm@11.0.8 --activate
WORKDIR /app
# Lockfile and the allowBuilds approvals first, so a source change does not re-resolve.
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
# The RDKit wasm reaches public/ through a postinstall hook that is `|| true` --
# in a clean image that copy ran before public/ existed and failed silently,
# and every structure then renders blank. Copy it where a missing file is a
# build failure instead.
RUN cp node_modules/@rdkit/rdkit/dist/RDKit_minimal.wasm public/RDKit_minimal.wasm
ARG APP_VERSION=0.0.0+dev
ARG APP_GIT_SHA=unknown
ARG APP_BUILD_DATE=unknown
RUN pnpm build

FROM node:22-alpine AS runtime
ENV NODE_ENV=production PORT=3000 HOSTNAME=0.0.0.0
# Build identity is read at request time by /api/config, so it is baked as ENV.
ARG APP_VERSION=0.0.0+dev
ARG APP_GIT_SHA=unknown
ARG APP_BUILD_DATE=unknown
ENV APP_VERSION=$APP_VERSION APP_GIT_SHA=$APP_GIT_SHA APP_BUILD_DATE=$APP_BUILD_DATE
WORKDIR /app
RUN addgroup -S app && adduser -S app -G app
# Standalone output omits the two directories the server expects beside it.
COPY --from=build --chown=app:app /app/.next/standalone ./
COPY --from=build --chown=app:app /app/.next/static ./.next/static
COPY --from=build --chown=app:app /app/public ./public
USER app
EXPOSE 3000
CMD ["node", "server.js"]
```

- [ ] **Step 2: Add the Makefile target**

```makefile
image-frontend: ## Build the daikon-frontend:local image (Next standalone + RDKit wasm)
	docker build -f frontend/Dockerfile -t daikon-frontend:local \
		--build-arg APP_VERSION=$$(git describe --tags --always) \
		--build-arg APP_GIT_SHA=$$(git rev-parse --short HEAD) \
		--build-arg APP_BUILD_DATE=$$(date -u +%Y-%m-%dT%H:%M:%SZ) frontend
```

Add `image-frontend` to `.PHONY`.

- [ ] **Step 3: Build and boot it**

```bash
make image-frontend
docker run -d --rm --name fe-smoke -p 127.0.0.1:3100:3000 -e APP_API_BASE_URL=http://localhost:8002 daikon-frontend:local
sleep 3
curl -s -o /dev/null -w "config %{http_code}\n" http://127.0.0.1:3100/api/config
curl -s -o /dev/null -w "login %{http_code}\n" http://127.0.0.1:3100/login
curl -s -o /dev/null -w "wasm %{http_code} %{size_download}\n" http://127.0.0.1:3100/RDKit_minimal.wasm
docker stop fe-smoke
```

Expected: `config 200`, `login 200`, `wasm 200 6900000`-ish bytes.

- [ ] **Step 4: Commit**

```bash
git add frontend/Dockerfile frontend/.dockerignore Makefile
git commit -m "feat(frontend): production Dockerfile (standalone, static, public, RDKit wasm)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: The deployment stack

**Files:**
- Create: `deploy/compose.yml`, `deploy/Caddyfile`, `deploy/.env.example`, `deploy/README.md`
- Modify: `.gitignore` (add `deploy/.env`)

**Interfaces:**
- Consumes: `daikon-runner:cpu` (Task 3), `daikon-frontend:local` (Task 4), `/ready` (Task 8; until then the healthcheck uses `/health`).
- Produces: `docker compose -f deploy/compose.yml up -d` brings up caddy, postgres, migrate, api, frontend, runner-default.

- [ ] **Step 1: compose.yml**

```yaml
# Production stack for one host. See deploy/README.md for the runbook.
# Images come from CI (ghcr.io) or from `make image-runner-cpu image-frontend`.
services:
  caddy:
    image: caddy:2-alpine
    restart: unless-stopped
    ports: ["80:80", "443:443"]
    environment:
      STUDIO_DOMAIN: "${STUDIO_DOMAIN}"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    depends_on: [api, frontend]

  postgres:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_USER: studio
      POSTGRES_PASSWORD: "${POSTGRES_PASSWORD}"
      POSTGRES_DB: studio
    volumes: ["pgdata:/var/lib/postgresql/data"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U studio -d studio"]
      interval: 5s
      timeout: 3s
      retries: 20

  # One-shot. The API image carries alembic; nothing else runs migrations.
  migrate:
    image: "${STUDIO_API_IMAGE}"
    command: ["alembic", "upgrade", "head"]
    environment:
      STUDIO_DATABASE_URL: "postgresql+asyncpg://studio:${POSTGRES_PASSWORD}@postgres:5432/studio"
    depends_on:
      postgres: { condition: service_healthy }
    restart: "no"

  api:
    image: "${STUDIO_API_IMAGE}"
    restart: unless-stopped
    env_file: .env
    environment:
      STUDIO_DATABASE_URL: "postgresql+asyncpg://studio:${POSTGRES_PASSWORD}@postgres:5432/studio"
      STUDIO_BLOB_BASE_URL: "file:///data/blobs"
      # Set here, not in .env: a JSON list survives Compose's `environment:`
      # verbatim, while shell sourcing and docker --env-file disagree about quotes.
      STUDIO_CORS_ORIGINS: '["https://${STUDIO_DOMAIN}"]'
      STUDIO_LOG_FORMAT: json
    volumes: ["blobs:/data/blobs"]
    depends_on:
      migrate: { condition: service_completed_successfully }
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready', timeout=3)"]
      interval: 15s
      timeout: 5s
      retries: 5

  frontend:
    image: "${STUDIO_FRONTEND_IMAGE}"
    restart: unless-stopped
    env_file: .env
    environment:
      APP_URL: "https://${STUDIO_DOMAIN}"
      APP_API_BASE_URL: "https://${STUDIO_DOMAIN}"
      APP_ENV: production

  # The default-lane runner, on the same host. Talks to the API directly, so
  # the reverse-proxy slash-merging caveat in deploy/README.md does not apply.
  runner-default:
    image: "${STUDIO_API_IMAGE}"
    restart: unless-stopped
    command: ["python", "-m", "daikonstudio.infrastructure.runner"]
    environment:
      STUDIO_URL: "http://api:8000"
      STUDIO_RUNNER_TOKEN: "${STUDIO_RUNNER_TOKEN_DEFAULT}"
      STUDIO_LOG_FORMAT: json
      OMP_NUM_THREADS: "1"
    depends_on: [api]

volumes:
  pgdata: {}
  blobs: {}
  caddy_data: {}
  caddy_config: {}
```

- [ ] **Step 2: Caddyfile**

```
# Everything under /api/v1, plus the probes and the schema, is the FastAPI
# backend. /api/config and /api/auth/mint are Next.js route handlers and MUST
# reach the frontend, so the matcher is /api/v1/*, not /api/*.
{$STUDIO_DOMAIN} {
	encode zstd gzip
	# Upload cap is 100 MB in the API (MAX_UPLOAD_BYTES); the proxy refuses a
	# little above it so an oversize body never spools to the API's disk.
	request_body {
		max_size 110MB
	}
	@backend path /api/v1/* /health /ready /version /docs /openapi.json
	handle @backend {
		reverse_proxy api:8000
	}
	handle {
		reverse_proxy frontend:3000
	}
}
```

- [ ] **Step 3: .env.example**

```bash
# Copy to deploy/.env and fill in. Both prefixes live in one file on purpose:
# the API reads STUDIO_*, the frontend reads APP_*, and the Duar key is the same.
STUDIO_DOMAIN=studio.example.edu
POSTGRES_PASSWORD=change-me

# Images. CI publishes these on every push to main; or build locally with
# `make image-runner-cpu image-frontend` and use daikon-runner:cpu / daikon-frontend:local.
STUDIO_API_IMAGE=ghcr.io/sidxz/daikon-studio/api:latest
STUDIO_FRONTEND_IMAGE=ghcr.io/sidxz/daikon-studio/frontend:latest

# Duar identity service (shared realm). Register a Service App for this
# deployment in the Duar admin panel; the name here must match it exactly.
STUDIO_DUAR_URL=https://duar.orca-03.biobio.tamu.edu
STUDIO_DUAR_SERVICE_NAME=daikon-studio
STUDIO_DUAR_SERVICE_KEY=
STUDIO_IDP_AUDIENCE=   # the realm's Google OAuth client id
APP_DUAR_URL=https://duar.orca-03.biobio.tamu.edu
APP_DUAR_SERVICE_NAME=daikon-studio
APP_DUAR_SERVICE_KEY=  # same value as STUDIO_DUAR_SERVICE_KEY
APP_DUAR_GOOGLE_CLIENT_ID=  # same value as STUDIO_IDP_AUDIENCE

# Per-lane job deadlines in seconds (JSON). Unlisted lanes use STUDIO_WORKER_JOB_TIMEOUT (1800).
STUDIO_WORKER_JOB_TIMEOUT_BY_LANE={"gpu": 7200}

# Created in the UI (Runners > New runner, admin role) AFTER first boot; then
# `docker compose up -d runner-default`.
STUDIO_RUNNER_TOKEN_DEFAULT=
```

- [ ] **Step 4: README runbook**

Write `deploy/README.md` with these sections, each a numbered list of commands:
1. **Prerequisites**: a Linux host with Docker 24+, DNS A record for `STUDIO_DOMAIN`, ports 80/443 open, the Google OAuth client's authorised redirect URI includes `https://<domain>/auth/callback`, and the Duar Service App registered.
2. **First boot**: `cp .env.example .env`, fill it, `docker compose pull`, `docker compose up -d`, `docker compose logs -f migrate api` until `Application startup complete`, open `https://<domain>`.
3. **First runner**: sign in as an admin, Runners → New runner (lane `default`), paste the token into `STUDIO_RUNNER_TOKEN_DEFAULT`, `docker compose up -d runner-default`. For a GPU box elsewhere: `docker run -d --restart unless-stopped --gpus all -e STUDIO_URL=https://<domain> -e STUDIO_RUNNER_TOKEN=drt_… <gpu image>` and note the slash-merging caveat: prediction artifact reads send a full `file:///…` URI in the path; Caddy preserves it, nginx's default `merge_slashes on` does not.
4. **Upgrade**: `docker compose pull && docker compose up -d` (the `migrate` service re-runs; the API waits for it).
5. **Backup and restore**: `docker compose exec -T postgres pg_dump -U studio studio | gzip > studio-$(date +%F).sql.gz` and `docker run --rm -v deploy_blobs:/blobs -v $PWD:/out alpine tar czf /out/blobs-$(date +%F).tgz -C /blobs .`; restore is the inverse with `psql` and `tar xzf`. Nightly cron suggestion.
6. **Logs and health**: `docker compose logs -f api`, `curl -s https://<domain>/ready`, every API log line carries `request_id`; pass `X-Request-ID` from a browser repro to find it.
7. **What this stack does not do**: no GPU runner on the host, no object storage (set `STUDIO_BLOB_BASE_URL=s3://…` plus `STUDIO_BLOB_STORAGE_OPTIONS` to move blobs), single API process, `/docs` is public.

- [ ] **Step 5: Validate the compose file and ignore the env**

```bash
echo "deploy/.env" >> .gitignore
cd deploy && cp .env.example .env && docker compose -f compose.yml config >/dev/null && echo OK && rm .env
```

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add deploy .gitignore
git commit -m "feat(deploy): compose stack behind Caddy with migrate step, runbook

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Continuous integration

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:

jobs:
  backend:
    runs-on: ubuntu-latest
    defaults: { run: { working-directory: backend } }
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with: { enable-cache: true }
      - run: uv sync --frozen --extra s3
      - run: uv run ruff check src tests && uv run ruff format --check src tests
      - run: uv run mypy src
      - run: uv run lint-imports
      # testcontainers needs the runner's Docker daemon, which ubuntu-latest has.
      - run: env OMP_NUM_THREADS=1 uv run pytest -q

  frontend:
    runs-on: ubuntu-latest
    defaults: { run: { working-directory: frontend } }
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
        with: { version: 11.0.8 }
      - uses: actions/setup-node@v4
        with: { node-version: 22, cache: pnpm, cache-dependency-path: frontend/pnpm-lock.yaml }
      - run: pnpm install --frozen-lockfile
      - run: pnpm lint
      - run: pnpm exec tsc --noEmit
      - run: pnpm test
      - run: pnpm build

  images:
    runs-on: ubuntu-latest
    needs: [backend, frontend]
    permissions: { contents: read, packages: write }
    env:
      API_IMAGE: ghcr.io/${{ github.repository }}/api
      FRONTEND_IMAGE: ghcr.io/${{ github.repository }}/frontend
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - name: Build the API image
        uses: docker/build-push-action@v6
        with: { context: backend, file: backend/Dockerfile, load: true, tags: daikon-runner:cpu }
      # A green build says nothing about whether the image works (docs/roadmap.md).
      - name: Boot-check the API image
        run: docker run --rm -e STUDIO_DUAR_SERVICE_KEY=smoke -e STUDIO_IDP_AUDIENCE=smoke daikon-runner:cpu python -c "import daikonstudio.interface.app, daikonstudio.infrastructure.engines.registry as r; print(sorted(m.id for m in r.default_registry().manifests()))"
      - name: Build the frontend image
        uses: docker/build-push-action@v6
        with:
          context: frontend
          file: frontend/Dockerfile
          load: true
          tags: daikon-frontend:ci
          build-args: |
            APP_VERSION=${{ github.ref_name }}
            APP_GIT_SHA=${{ github.sha }}
            APP_BUILD_DATE=${{ github.event.head_commit.timestamp || github.event.pull_request.updated_at }}
      - name: Boot-check the frontend image
        run: |
          docker run -d --name fe -p 3000:3000 daikon-frontend:ci && sleep 4
          curl -fsS http://localhost:3000/api/config >/dev/null && curl -fsSI http://localhost:3000/RDKit_minimal.wasm >/dev/null
      - uses: docker/login-action@v3
        if: github.event_name == 'push'
        with: { registry: ghcr.io, username: "${{ github.actor }}", password: "${{ secrets.GITHUB_TOKEN }}" }
      - name: Push
        if: github.event_name == 'push'
        run: |
          docker tag daikon-runner:cpu $API_IMAGE:latest && docker tag daikon-runner:cpu $API_IMAGE:${{ github.sha }}
          docker tag daikon-frontend:ci $FRONTEND_IMAGE:latest && docker tag daikon-frontend:ci $FRONTEND_IMAGE:${{ github.sha }}
          docker push --all-tags $API_IMAGE && docker push --all-tags $FRONTEND_IMAGE
```

- [ ] **Step 2: Validate the YAML parses**

Run: `python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"` (or `uv run --with pyyaml python -c …` from `backend/`).
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: lint, test, build and boot-check both images; publish to ghcr on main

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Logging that exists

**Files:**
- Create: `backend/src/daikonstudio/logging.py`
- Modify: `backend/src/daikonstudio/settings.py` (two fields)
- Modify: `backend/src/daikonstudio/interface/app.py:24-26`
- Modify: `backend/src/daikonstudio/infrastructure/runner/agent.py:60-65,135-139`
- Modify: `backend/src/daikonstudio/infrastructure/jobs.py:223-228`
- Test: `backend/tests/unit/test_logging.py`

**Interfaces:**
- Produces: `configure_logging(*, level: str = "INFO", fmt: str = "console") -> None`; `DropNoisyAccess(logging.Filter)`; `Settings.log_level`, `Settings.log_format`; `AgentSettings.log_level`, `AgentSettings.log_format`.

- [ ] **Step 1: Failing tests**

```python
"""Logging exists, renders JSON on demand, and stays quiet about runner polls."""

import json
import logging

from daikonstudio.logging import DropNoisyAccess, configure_logging


def _access(message: str) -> logging.LogRecord:
    return logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, message, None, None)


def test_runner_polls_and_probes_are_dropped_from_the_access_log():
    noisy = DropNoisyAccess()
    assert not noisy.filter(_access('127.0.0.1:1 - "POST /api/v1/runner/claim HTTP/1.1" 204 No Content'))
    assert not noisy.filter(_access('127.0.0.1:1 - "GET /health HTTP/1.1" 200 OK'))
    assert noisy.filter(_access('127.0.0.1:1 - "POST /api/v1/runs HTTP/1.1" 202 Accepted'))
    # A failing poll is news.
    assert noisy.filter(_access('127.0.0.1:1 - "POST /api/v1/runner/claim HTTP/1.1" 500 Internal Server Error'))


def test_json_format_renders_stdlib_records_as_json(capsys):
    configure_logging(level="INFO", fmt="json")
    logging.getLogger("daikonstudio.test").warning("hello %s", "world", extra={})
    line = capsys.readouterr().err.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["event"] == "hello world"
    assert record["level"] == "warning"
    assert record["logger"] == "daikonstudio.test"


def test_info_is_visible_once_configured(capsys):
    configure_logging(level="INFO", fmt="console")
    logging.getLogger("daikonstudio.test").info("realm scope active")
    assert "realm scope active" in capsys.readouterr().err
```

- [ ] **Step 2: Run, expect ImportError**

Run: `cd backend && uv run pytest tests/unit/test_logging.py -q`

- [ ] **Step 3: Implement `logging.py`**

```python
"""The one place logging is configured, for the API process and the runner agent.

Nothing in this app configured logging before 2026-10-02: the root logger had no
handler, so Python's lastResort fallback emitted WARNING and above to stderr and
dropped INFO -- which is why the Duar realm-scope line never appeared and the
auth wedge took a long investigation. stdlib and structlog records both render
through one structlog formatter, as console text for a terminal or JSON for a
log collector.
"""

from __future__ import annotations

import logging
import sys

import structlog

#: Paths whose successful access lines say nothing: two runners polling every
#: 3 s is 40 lines a minute. Failures on these paths still log.
NOISY_PATHS = ("/api/v1/runner/claim", "/health", "/ready")


class DropNoisyAccess(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        quiet_path = any(f" {path} " in message for path in NOISY_PATHS)
        success = '" 200 ' in message or '" 204 ' in message
        return not (quiet_path and success)


def configure_logging(*, level: str = "INFO", fmt: str = "console") -> None:
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    renderer = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        )
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # Uvicorn installs its own handlers before the app imports; route them
    # through ours so every line has one shape, and mute the poll chatter.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv = logging.getLogger(name)
        uv.handlers[:] = []
        uv.propagate = True
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, DropNoisyAccess) for f in access.filters):
        access.addFilter(DropNoisyAccess())
```

- [ ] **Step 4: Settings and wiring**

`settings.py`, after `service_name`:

```python
    # "console" for a terminal, "json" for a log collector. Level names are
    # stdlib's. Both read once at boot by `daikonstudio.logging.configure_logging`.
    log_level: str = "INFO"
    log_format: str = "console"
```

`interface/app.py`, `create_app()` first two lines:

```python
    settings = Settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)
```

with `from daikonstudio.logging import configure_logging`.

`infrastructure/runner/agent.py`: add to `AgentSettings`:

```python
    log_level: str = "INFO"
    log_format: str = "console"
```

and in `main()` right after `settings = AgentSettings()`: `configure_logging(level=settings.log_level, fmt=settings.log_format)`.

`infrastructure/jobs.py`: add `import structlog` and `_logger = structlog.get_logger(__name__)`; replace `InlineEnqueuer.enqueue`'s body:

```python
        try:
            await run_job(self._ctx, run_id)
        except (Exception, SystemExit):
            # run_job already persisted FAILED; this is so the operator sees it,
            # exactly as the runner agent logs the same failure.
            _logger.exception("inline job failed", run_id=str(run_id))
```

Update the docstring sentence "Swallowing the exception after `run_job()` records it" to "Logging, not re-raising, after `run_job()` records it".

- [ ] **Step 5: Run tests, lint**

Run: `cd backend && uv run pytest tests/unit/test_logging.py tests/unit/execution/test_jobs.py -q && cd .. && make lint`
Expected: pass; lint clean. Then `make dev-be` and check `.logs/backend.log` shows the `duar` INFO line and no `runner/claim` 204 lines.

- [ ] **Step 6: Commit**

```bash
git add backend/src/daikonstudio/logging.py backend/src/daikonstudio/settings.py backend/src/daikonstudio/interface/app.py backend/src/daikonstudio/infrastructure/runner/agent.py backend/src/daikonstudio/infrastructure/jobs.py backend/tests/unit/test_logging.py
git commit -m "feat(ops): configure logging (console/json), mute runner polls, log inline job failures

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Request ids, a JSON 500, and a readiness probe

**Files:**
- Create: `backend/src/daikonstudio/interface/middleware.py`
- Modify: `backend/src/daikonstudio/interface/error_handlers.py:60-66`
- Modify: `backend/src/daikonstudio/interface/app.py` (middleware order, `/ready`, exclude path)
- Modify: `backend/src/daikonstudio/logging.py` (`/ready` already in `NOISY_PATHS`)
- Test: `backend/tests/api/test_health.py`, `backend/tests/unit/test_middleware.py`

**Interfaces:**
- Produces: response header `X-Request-ID` on every response; `GET /ready` → 200 `{"status":"ready"}` or 503; unhandled exceptions → `{"error":"InternalError","message":…,"request_id":…}` with CORS headers for an allowed Origin. `register_error_handlers(app, cors_origins: list[str])`. `check_database(sessions) -> str | None`.

- [ ] **Step 1: Failing tests**

`tests/api/test_health.py`, append:

```python
import httpx
from sqlalchemy.ext.asyncio import async_sessionmaker

from daikonstudio.interface.app import check_database


async def test_every_response_carries_a_request_id(client):
    response = await client.get("/health")
    assert response.headers["x-request-id"]


async def test_a_supplied_request_id_is_echoed(client):
    response = await client.get("/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["x-request-id"] == "abc123"


async def test_ready_is_200_when_the_database_answers(client):
    response = await client.get("/ready")
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "ready"}


async def test_check_database_names_the_failure_without_raising():
    broken = async_sessionmaker(bind=None)  # no engine: using it raises

    assert await check_database(broken) is not None


async def test_an_unhandled_error_is_a_json_500_with_cors_and_request_id(app, signing_key, workspace_id):
    @app.get("/__boom")
    async def boom() -> None:
        raise RuntimeError("kaboom")

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as anonymous:
        response = await anonymous.get(
            "/__boom", headers={"Origin": "http://localhost:3003", "X-Request-ID": "r-1"}
        )
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "InternalError"
    assert body["request_id"] == "r-1"
    assert "kaboom" not in response.text
    assert response.headers["access-control-allow-origin"] == "http://localhost:3003"
```

Note: `/__boom` sits behind Duar; the api conftest's `app` fixture overrides auth so `anonymous` reaches the route only if the path is excluded — instead register the route and add `"/__boom"` to nothing: use the authenticated `client` fixture's headers (`auth_headers(signing_key, workspace_id)` from conftest) on the anonymous transport. Adjust when implementing to whichever helper the conftest exposes.

- [ ] **Step 2: Run, expect failures**

Run: `cd backend && uv run pytest tests/api/test_health.py -q`

- [ ] **Step 3: Middleware**

```python
"""Request correlation. One id per request, generated or honoured from
`X-Request-ID`, bound into structlog's context so every log line emitted while
serving the request carries it, and echoed on the response so a browser repro
can be matched to the server log."""

from __future__ import annotations

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
```

- [ ] **Step 4: Error handler and readiness**

`error_handlers.py`: change the signature to `register_error_handlers(app: FastAPI, *, cors_origins: list[str] | None = None)` and add inside it:

```python
    allowed = set(cors_origins or [])
    logger = structlog.get_logger(__name__)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.exception("unhandled error", request_id=request_id)
        headers: dict[str, str] = {}
        origin = request.headers.get("origin")
        # ServerErrorMiddleware sits outside CORSMiddleware, so a 500 would
        # otherwise reach the browser without CORS headers and read as a network
        # error instead of a server error.
        if origin in allowed:
            headers = {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true",
                "Vary": "Origin",
            }
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalError",
                "message": "Something went wrong on the server",
                "request_id": request_id,
            },
            headers=headers,
        )
```

`app.py`:

```python
async def check_database(sessions: async_sessionmaker[AsyncSession]) -> str | None:
    """None when `SELECT 1` answers within 3 s, else the failure's class name."""
    try:
        async with asyncio.timeout(3), sessions() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 -- a probe reports, never raises
        return type(exc).__name__
    return None
```

In `create_app()`: `app.add_middleware(RequestIdMiddleware)` **after** `CORSMiddleware` (LIFO: outermost), `register_error_handlers(app, cors_origins=settings.cors_origins)`, add `"/ready"` to `exclude_paths`, and:

```python
    @app.get("/ready")
    async def ready() -> JSONResponse:
        failure = await check_database(app.state.container[async_sessionmaker])
        if failure:
            return JSONResponse(status_code=503, content={"status": "unavailable", "detail": failure})
        return JSONResponse({"status": "ready"})
```

- [ ] **Step 5: Run tests and lint**

Run: `cd backend && uv run pytest tests/api/test_health.py -q && make -C .. lint`

- [ ] **Step 6: Commit**

```bash
git add backend/src/daikonstudio/interface backend/tests/api/test_health.py
git commit -m "feat(ops): request ids, JSON 500s that survive CORS, and GET /ready

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Failed runs say something a user can read

**Files:**
- Create: `backend/src/daikonstudio/application/execution/failure_message.py`
- Modify: `backend/src/daikonstudio/infrastructure/jobs.py:166-169`
- Modify: `backend/src/daikonstudio/application/execution/train_protocol.py:665-676`
- Test: `backend/tests/unit/execution/test_failure_message.py`, extend `tests/unit/execution/test_jobs.py`

**Interfaces:**
- Produces: `user_facing_error(exc: BaseException) -> str`.

- [ ] **Step 1: Failing tests**

```python
from daikonstudio.application.execution.failure_message import user_facing_error
from daikonstudio.domain.shared.errors import ValidationError


def test_domain_errors_pass_through_with_their_detail():
    error = ValidationError("conditions invalid", detail="n_estimators above maximum 2000")
    assert user_facing_error(error) == "conditions invalid (n_estimators above maximum 2000)"


def test_library_value_errors_keep_their_text():
    assert user_facing_error(ValueError("Input y contains NaN.")) == "ValueError: Input y contains NaN."


def test_everything_else_is_reduced_to_its_class():
    message = user_facing_error(FileNotFoundError("/data/blobs/ws/protocols/x/model.joblib"))
    assert "/data/blobs" not in message
    assert "FileNotFoundError" in message
```

And in `test_jobs.py::test_run_job_records_failure_and_reraises`, assert `run.error_message == "ValueError: engine exploded"`.

- [ ] **Step 2: Implement**

```python
"""What a workspace member reads on a failed run.

`repr(exc)` carried blob paths and internal URLs to every viewer of the run.
Domain errors are written for users and pass through; `ValueError`/`TypeError`
from a library name the data problem (an `Input y contains NaN` is actionable)
and keep their text; everything else is reduced to its class name, with the
full traceback in the server log.
"""

from __future__ import annotations

from daikonstudio.domain.shared.errors import DomainError

_MAX_LENGTH = 500


def user_facing_error(exc: BaseException) -> str:
    if isinstance(exc, DomainError):
        return f"{exc.message} ({exc.detail})" if exc.detail else exc.message
    if isinstance(exc, ValueError | TypeError):
        return f"{type(exc).__name__}: {str(exc)[:_MAX_LENGTH]}"
    return f"Unexpected {type(exc).__name__}; the server log has the details"
```

`jobs.py:167`: `run.fail(user_facing_error(exc))`. `train_protocol.py:676`: `return None, user_facing_error(exc), None`.

- [ ] **Step 3: Run tests, lint, commit**

```bash
git commit -am "fix(runs): user-facing failure messages instead of repr(exc)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Dataset validation rejects what training would reject

**Files:**
- Modify: `backend/src/daikonstudio/application/data/prepare_frame.py`
- Modify: `backend/src/daikonstudio/application/data/create_dataset.py:123-126,141-143`
- Modify: `backend/src/daikonstudio/application/execution/predict_with_protocol.py:270-273`
- Test: `backend/tests/unit/data/test_validation.py`

**Interfaces:**
- Produces: `read_csv_upload(raw: bytes) -> pl.DataFrame` (raises `ValidationError`; strips a header BOM); `prepare_frame` now also emits `InvalidRow`s with reasons `"empty target value"`, `"target is not a number: '<raw>'"`, `"binary target must be 0 or 1, got '<raw>'"`, and returns the target column as Float64 (numeric) or Int64 (binary).

- [ ] **Step 1: Failing tests** (append to `test_validation.py`)

```python
import io


def test_a_bom_on_the_first_header_is_stripped():
    from daikonstudio.application.data.prepare_frame import read_csv_upload

    frame = read_csv_upload("﻿smiles,y\nCCO,1.0\n".encode())
    assert frame.columns == ["smiles", "y"]


def test_null_numeric_targets_are_invalid_rows_not_training_failures():
    frame = pl.DataFrame({"smiles": ["CCO", "CCC"], "y": [1.0, None]})
    prepared, report = prepare_frame(frame, "smiles", NUMERIC, NORMALIZER)
    assert prepared.height == 1
    assert report.invalid == [InvalidRow(row_number=2, value="", reason="empty target value")]


def test_non_numeric_target_values_are_invalid_rows_not_a_crash():
    frame = pl.DataFrame({"smiles": ["CCO", "CCC", "CCN"], "y": ["1.5", "NA", "<10"]})
    prepared, report = prepare_frame(frame, "smiles", NUMERIC, NORMALIZER)
    assert prepared["y"].dtype == pl.Float64
    assert prepared["y"].to_list() == [1.5]
    assert [r.row_number for r in report.invalid] == [2, 3]
    assert report.invalid[0].reason == "target is not a number: 'NA'"


def test_binary_targets_must_be_zero_or_one():
    frame = pl.DataFrame({"smiles": ["CCO", "CCC", "CCN"], "y": ["1", "active", "2"]})
    prepared, report = prepare_frame(frame, "smiles", BINARY, NORMALIZER)
    assert prepared["y"].to_list() == [1]
    assert [r.reason for r in report.invalid] == [
        "binary target must be 0 or 1, got 'active'",
        "binary target must be 0 or 1, got '2'",
    ]
```

(`InvalidRow` import from `daikonstudio.domain.data.validation`.)

- [ ] **Step 2: Run, expect failures**

- [ ] **Step 3: Implement**

In `prepare_frame.py` add:

```python
def read_csv_upload(raw: bytes) -> pl.DataFrame:
    """Every CSV this app accepts comes through here. A UTF-8 BOM on the first
    header is stripped: Excel writes one, and `\\ufeffsmiles` is not a column a
    scientist can select."""
    try:
        frame = pl.read_csv(io.BytesIO(raw))
    except pl.exceptions.PolarsError as error:
        raise ValidationError(f"The uploaded file is not readable as CSV: {error}") from error
    return frame.rename({c: c.lstrip("﻿") for c in frame.columns if c.startswith("﻿")})


def _validate_target(
    frame: pl.DataFrame, target: TargetSpec, row_numbers: list[int]
) -> tuple[pl.DataFrame, list[int], list[InvalidRow]]:
    """Rows whose target cannot be trained on, rejected here with their row
    numbers rather than as `Input y contains NaN` minutes later in a worker.
    Returns the frame with the target cast (Float64 numeric, Int64 binary)."""
    raw = frame[target.column]
    text = raw.cast(pl.Utf8, strict=False).fill_null("").str.strip_chars()
    numeric = raw.cast(pl.Float64, strict=False) if raw.dtype != pl.Utf8 else raw.str.strip_chars().cast(pl.Float64, strict=False)
    empty = text == ""
    if target.kind is TargetKind.BINARY:
        ok = numeric.is_in([0.0, 1.0]) & ~empty
        reason = "binary target must be 0 or 1, got '{raw}'"
        cast_to: pl.DataType = pl.Int64()
    else:
        ok = numeric.is_not_null() & ~empty
        reason = "target is not a number: '{raw}'"
        cast_to = pl.Float64()
    invalid = [
        InvalidRow(
            row_number=row_numbers[i],
            value=text[i],
            reason="empty target value" if empty[i] else reason.format(raw=text[i]),
        )
        for i in range(frame.height)
        if not ok[i]
    ]
    kept = frame.filter(ok).with_columns(numeric.filter(ok).cast(cast_to).alias(target.column))
    return kept, [n for n, keep in zip(row_numbers, ok, strict=True) if keep], invalid
```

Call it in `prepare_frame` right after `valid_frame`/`row_numbers` are built (line 67) and before `valid_rows = valid_frame.height`:

```python
    valid_frame, row_numbers, bad_targets = _validate_target(valid_frame, target, row_numbers)
    invalid.extend(bad_targets)
    invalid.sort(key=lambda row: row.row_number)
```

`create_dataset.py:123-126`: replace the `try: pl.read_csv …` block with `frame = read_csv_upload(self._store.get_bytes(key))` inside a `try/except ValidationError as error: return Failure(error)`. Wrap the `prepare_frame(...)` call (141-143) in `try/except pl.exceptions.PolarsError as error: return Failure(ValidationError(f"The file could not be interpreted: {error}"))`.

`predict_with_protocol.py:270-273`: `frame = read_csv_upload(raw)` (import from `daikonstudio.application.data.prepare_frame`).

- [ ] **Step 4: Run the data and api suites, lint, commit**

Run: `cd backend && uv run pytest tests/unit/data tests/api/test_datasets.py tests/integration/test_prediction_cache.py -q`

```bash
git commit -am "fix(datasets): reject empty, non-numeric and non-binary targets at upload; strip header BOM

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Compound identifiers and input rows survive prediction

**Files:**
- Modify: `backend/src/daikonstudio/domain/data/target.py:31-33`
- Modify: `backend/src/daikonstudio/domain/execution/run.py` (new `record_prediction_counts`)
- Modify: `backend/src/daikonstudio/application/execution/predict_with_protocol.py` (command, cache key, `RunPrediction`, `PredictionRow`, `GetPredictionResults`)
- Modify: `backend/src/daikonstudio/interface/routes/runs.py` (`PredictBody.id_column`, `PredictionResponse`, `RunResponse.metrics`)
- Modify: `backend/src/daikonstudio/application/data/export_collection.py` (`_final_labels`, SDF tags)
- Test: `backend/tests/integration/test_prediction_cache.py`, `backend/tests/api/test_runs.py`

**Interfaces:**
- Produces: results Parquet columns `structure, input_row (Int64), compound_id (Utf8, optional), <readouts>, uncertainty, applicability`; `PredictionResponse.input_row: int | None`, `.compound_id: str | None`; `PredictBody.id_column: str | None`; `RunResponse.metrics: dict | None`, for prediction runs `{"uploaded_rows": m, "scored_rows": n}`; `Run.record_prediction_counts(*, uploaded_rows: int, scored_rows: int) -> None`.

- [ ] **Step 1: Failing tests**

In `tests/integration/test_prediction_cache.py` extend `test_predictions_carry_structure_readouts_uncertainty_and_applicability` with:

```python
    # Row 3 ("not-a-molecule") was dropped; the gap in input_row says so.
    assert frame["input_row"].to_list() == [1, 2, 4]
    assert run.metrics == {"uploaded_rows": 4, "scored_rows": 3}
```

and add:

```python
async def test_a_blank_identifier_is_null_and_duplicates_both_survive(
    studio: Studio, published_protocol: InSilicoProtocol
) -> None:
    csv = b"name,smiles\nCPD-1,CCO\n,CCC\nCPD-1,c1ccccc1\n"
    upload_ref = await studio.upload(csv)
    run = await studio.wait(await studio.predict(published_protocol.id, upload_ref, id_column="name"))
    frame = studio.results_frame(run)
    assert frame["compound_id"].to_list() == ["CPD-1", None, "CPD-1"]


async def test_an_identifier_column_that_does_not_exist_fails_the_run_clearly(
    studio: Studio, published_protocol: InSilicoProtocol, upload_ref: str
) -> None:
    run = await studio.wait(await studio.predict(published_protocol.id, upload_ref, id_column="nope"))
    assert run.status is RunStatus.FAILED
    assert "nope" in (run.error_message or "")
```

(Extend the `Studio.predict` helper in the integration conftest with `id_column: str | None = None`.)

In `tests/api/test_runs.py` add a test that `POST /runs` with `"id_column": "name"` is 202 and that `GET /runs/{id}/results` items carry `compound_id` and `input_row`.

- [ ] **Step 2: Run, expect failures**

- [ ] **Step 3: Implement**

`target.py`: `RESERVED_TARGET_COLUMNS = frozenset({"structure", "uncertainty", "applicability", "generation_method", "row_id", "split", "input_row", "compound_id"})`.

`run.py`, after `record_metrics`:

```python
    def record_prediction_counts(self, *, uploaded_rows: int, scored_rows: int) -> None:
        """How many rows the upload held and how many were scored. The
        difference is the structures that did not parse, and a scientist must
        be told that number rather than left to notice 9,970 where 9,975 went in."""
        self.metrics = {"uploaded_rows": uploaded_rows, "scored_rows": scored_rows}
        self._touch()
```

`predict_with_protocol.py`:
- `PredictWithProtocolCommand.id_column: str | None = None`; `to_params` adds `"id_column": self.id_column`; `from_params` reads `params.get("id_column")`.
- `compute_cache_key(..., id_column=command.id_column)`.
- `RunPrediction.__call__`, after `valid_frame` is built:

```python
        keep = [ok for ok in is_valid.to_list()]
        input_rows = [index + 1 for index, ok in enumerate(keep) if ok]
        compound_ids: list[str | None] | None = None
        if command.id_column is not None:
            if command.id_column not in frame.columns:
                raise ValidationError(
                    f"Identifier column '{command.id_column}' not present in the uploaded "
                    f"file: available columns: {', '.join(frame.columns)}"
                )
            raw_ids = frame[command.id_column].cast(pl.Utf8, strict=False).to_list()
            compound_ids = [
                None if value is None or not str(value).strip() else str(value).strip()
                for value, ok in zip(raw_ids, keep, strict=True)
                if ok
            ]
```

and when assembling `columns`, after `"structure"`:

```python
        columns["input_row"] = pl.Series(input_rows, dtype=pl.Int64)
        if compound_ids is not None:
            columns["compound_id"] = pl.Series(compound_ids, dtype=pl.Utf8)
```

and before returning: `run.record_prediction_counts(uploaded_rows=frame.height, scored_rows=valid_frame.height)` (the aggregate rides out on `run_job`'s `succeed()` + `update()`, exactly as `record_metrics` does for training).

- `PredictionRow`: add `input_row: int | None`, `compound_id: str | None`. In `GetPredictionResults`: `input_row=row.get("input_row")`, `compound_id=row.get("compound_id")` (older results files lack both; `.get` keeps them readable).

`routes/runs.py`: `PredictBody.id_column: str | None = None` → command; `PredictionResponse.input_row: int | None`, `.compound_id: str | None`; `RunResponse.metrics: dict[str, Any] | None` from `run.metrics`.

`export_collection.py`: `_final_labels` CSV branch returns `[*_csv_rename(readouts).values(), "input_row", "compound_id", "generation_method"]` only when present; simplest: leave the frame's extra columns untouched in `_render_csv` (they already pass through `rename`) and in `_render_sdf` write `input_row` and `compound_id` as SD tags when the frame has them (read `_render_sdf` first; add the two tags beside `generation_method`).

- [ ] **Step 4: Regenerate the contract**

Run: `make generate-api` (frontend types pick up `id_column`, `compound_id`, `input_row`, `metrics`).

- [ ] **Step 5: Tests, lint, commit**

Run: `cd backend && uv run pytest tests/integration/test_prediction_cache.py tests/api/test_runs.py tests/api/test_collections.py tests/unit -q`

```bash
git add -A && git commit -m "feat(predictions): carry input_row and an optional compound_id; record uploaded vs scored counts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: Deadlines that bind, and lanes a client can see

**Files:**
- Modify: `backend/src/daikonstudio/settings.py` (`worker_job_timeout_by_lane`)
- Modify: `backend/src/daikonstudio/application/execution/claim_run.py`
- Modify: `backend/src/daikonstudio/infrastructure/di/container.py:176-186`
- Modify: `backend/src/daikonstudio/domain/execution/run.py` (`lane` attribute)
- Modify: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/execution/repository.py:32-45` (`_to_domain` maps `lane`)
- Modify: `backend/src/daikonstudio/application/execution/train_protocol.py:730-784` (`_check_deadline`, `_progress`)
- Modify: `backend/src/daikonstudio/interface/routes/engines.py` (`lane`), `routes/runs.py` (`lane`)
- Test: `backend/tests/unit/execution/test_reporter.py`, `backend/tests/unit/runners/test_claim_deadline.py`, `backend/tests/api/test_engines.py`

**Interfaces:**
- Produces: `Settings.worker_job_timeout_by_lane: dict[str, int]`; `ClaimRun(..., deadline_seconds: int, deadline_by_lane: dict[str, int])`; `Run.lane: str | None` (read-only, informational); `EngineManifestResponse.lane: str`; `RunResponse.lane: str | None`.

- [ ] **Step 1: Failing tests**

`tests/unit/execution/test_reporter.py`, append:

```python
async def test_progress_between_fits_honours_the_deadline():
    """Tree and GP engines never call ctx.report, so the only deadline check
    they can hit is the one between fits."""
    training = RunTraining(*([None] * 6), deadline_seconds=1)
    training._deadline_at = time.monotonic() - 1
    run = Run(kind=RunKind.TRAINING, workspace_id=uuid.uuid4(), requested_by=uuid.uuid4(), cache_key="k")
    run.start()
    with pytest.raises(RunInterrupted) as raised:
        await training._progress(run, 0.6, "training baseline")
    assert raised.value.cancelled is False
    assert "deadline" in raised.value.reason
```

`tests/unit/runners/test_claim_deadline.py`:

```python
async def test_the_claimed_runs_lane_picks_its_deadline():
    run = Run(kind=RunKind.TRAINING, workspace_id=uuid.uuid4(), requested_by=uuid.uuid4(), cache_key="k", lane="gpu")

    class Queue:
        async def sweep(self, *, max_attempts): ...
        async def claim_next(self, **_): return run.id

    class Runs:
        async def get_by_id(self, run_id): return run

    claim = ClaimRun(Queue(), Runs(), lease_seconds=600, max_active_per_workspace=10, max_attempts=3,
                     deadline_seconds=1800, deadline_by_lane={"gpu": 7200})
    runner = Runner(name="g", lanes=("gpu",), token_hash="h")
    claimed = (await claim(runner=runner)).unwrap()
    assert claimed == (run, 7200, 600)
```

`tests/api/test_engines.py`: assert every item has `"lane"` and `chemprop-dmpnn` has `lane == "gpu"`.

- [ ] **Step 2: Implement**

`settings.py`: `worker_job_timeout_by_lane: dict[str, int] = {}` with a comment pointing at `STUDIO_WORKER_JOB_TIMEOUT_BY_LANE='{"gpu": 7200}'`.

`run.py`: `lane: str | None = None` in `__init__` → `self.lane = lane` with the comment "Queue column, read-only here: set by the enqueuer, never persisted by `update()`." `repository.py::_to_domain`: `lane=model.lane`.

`claim_run.py`: constructor gains `deadline_by_lane: dict[str, int] | None = None`; the return becomes `self._deadline_by_lane.get(run.lane or DEFAULT_LANE, self._deadline_seconds)`. `container.py`: pass `deadline_by_lane=resolved.worker_job_timeout_by_lane`.

`train_protocol.py`: factor the reporter's deadline check into

```python
    def _check_deadline(self) -> None:
        if self._deadline_at is not None and time.monotonic() > self._deadline_at:
            raise RunInterrupted(
                f"exceeded the {self._deadline_seconds:.0f}s job deadline; raise "
                "STUDIO_WORKER_JOB_TIMEOUT, or this lane's entry in "
                "STUDIO_WORKER_JOB_TIMEOUT_BY_LANE, if the work is legitimate",
                cancelled=False,
            )
```

called from `report()` (replacing lines 754-760) and as the first line of `_progress`.

`routes/engines.py`: `lane: str` from `manifest.lane`. `routes/runs.py`: `lane: str | None` from `run.lane`.

- [ ] **Step 3: `make generate-api`, tests, lint, commit**

```bash
git add -A && git commit -m "feat(runs): per-lane job deadlines, deadline checked between fits, lane on engine and run responses

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: Retry a failed or cancelled run

**Files:**
- Modify: `backend/src/daikonstudio/domain/execution/run.py` (`retry`)
- Modify: `backend/src/daikonstudio/application/execution/train_protocol.py` (`training_lane` helper; `TrainProtocol` uses it)
- Create: `backend/src/daikonstudio/application/execution/retry_run.py`
- Modify: `backend/src/daikonstudio/infrastructure/di/container.py` (define `RetryRun`)
- Modify: `backend/src/daikonstudio/interface/routes/runs.py` (`POST /{run_id}/retry`)
- Test: `backend/tests/unit/execution/test_run.py`, `backend/tests/api/test_runs.py`

**Interfaces:**
- Produces: `Run.retry() -> None` (`failed|cancelled -> pending`, clears progress/phase/error_message); `training_lane(engines: EngineRegistry, engine_id: str, baseline_engine_id: str | None) -> str`; `RetryRun(runs, protocols, enqueuer, engines)` with `RetryRunCommand(run_id)`; route `POST /api/v1/runs/{run_id}/retry` → 204.

- [ ] **Step 1: Failing tests**

`test_run.py`:

```python
def test_retry_returns_a_failed_run_to_pending_and_clears_the_failure():
    run = _pending()
    run.start()
    run.report_progress(0.4, phase="training")
    run.fail("boom")
    run.retry()
    assert (run.status, run.progress, run.phase, run.error_message) == (RunStatus.PENDING, 0.0, None, None)


def test_retry_is_allowed_from_cancelled_too():
    run = _pending()
    run.cancel()
    run.retry()
    assert run.status is RunStatus.PENDING


@pytest.mark.parametrize("prepare", [lambda r: None, lambda r: r.start(), lambda r: (r.start(), r.succeed("uri"))])
def test_retry_refuses_pending_running_and_ready(prepare):
    """`running` is the one that matters: a crashed worker leaves a run RUNNING, and
    a retry from there would start a second fit beside one that may still be alive."""
    run = _pending()
    prepare(run)
    with pytest.raises(ConflictError):
        run.retry()
```

`tests/api/test_runs.py`:

```python
async def test_retrying_a_failed_prediction_reenqueues_and_runs_it(
    client, session_factory, workspace_id, published_protocol_id, prediction_upload_ref
):
    command = PredictWithProtocolCommand(
        protocol_id=uuid.UUID(published_protocol_id), upload_ref=prediction_upload_ref, structure_column="smiles"
    )
    run = Run(kind=RunKind.PREDICTION, workspace_id=workspace_id, requested_by=uuid.uuid4(),
              cache_key="retry-me", params=command.to_params(), protocol_id=uuid.UUID(published_protocol_id))
    run.start()
    run.fail("the runner died")
    await SqlAlchemyRunRepository(session_factory).add(run)

    response = await client.post(f"/api/v1/runs/{run.id}/retry")
    assert response.status_code == 204, response.text
    polled = (await client.get(f"/api/v1/runs/{run.id}")).json()
    # Inline jobs run inside the request, so the retried run has already finished.
    assert polled["status"] == "ready"
    assert polled["error_message"] is None


async def test_retrying_a_ready_run_is_a_409(client, published_protocol_id, prediction_upload_ref):
    run_id = (await _predict(client, published_protocol_id, prediction_upload_ref)).json()["id"]
    assert (await client.post(f"/api/v1/runs/{run_id}/retry")).status_code == 409


async def test_viewer_cannot_retry(viewer_client, client, session_factory, workspace_id):
    run = Run(kind=RunKind.PREDICTION, workspace_id=workspace_id, requested_by=uuid.uuid4(), cache_key="k4")
    run.start(); run.fail("x")
    await SqlAlchemyRunRepository(session_factory).add(run)
    assert (await viewer_client.post(f"/api/v1/runs/{run.id}/retry")).status_code == 403
```

- [ ] **Step 2: Implement**

`run.py`:

```python
    def retry(self) -> None:
        """`failed -> pending` and `cancelled -> pending`, the only edges out of a
        terminal status. `params` is write-once, so the re-enqueued job re-reads the
        same instructions; nothing is rebuilt. `running` is excluded on purpose: a
        crashed worker leaves a run RUNNING, and a retry from there would start a
        second fit beside one that may still be alive. Cancel first, then retry."""
        if self.status not in {RunStatus.FAILED, RunStatus.CANCELLED}:
            raise ConflictError(f"Cannot retry run '{self.id}' in status '{self.status}'")
        self.status = RunStatus.PENDING
        self.progress = 0.0
        self.phase = None
        self.error_message = None
        self._touch()
```

`train_protocol.py`:

```python
def training_lane(engines: EngineRegistry, engine_id: str, baseline_engine_id: str | None) -> str:
    """The lane a training Run needs: both the chosen engine and its baseline fit
    inside one job. Shared by enqueue and retry so the rule has one home."""
    engine = engines.get(engine_id)
    baseline = engines.get(baseline_engine_id) if baseline_engine_id else engines.baseline()
    return lane_for(engine.manifest(), baseline.manifest())
```

and `TrainProtocol.__call__` line 381 becomes `lane=training_lane(self._engines, command.engine_id, command.baseline_engine_id)`.

`retry_run.py`:

```python
"""Re-execute a failed or cancelled Run in place. Spec: 2026-08-04-retry-a-failed-run-design."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.engines.registry import EngineRegistry, UnknownEngineError
from daikonstudio.application.execution.enqueue import JobEnqueuer
from daikonstudio.application.execution.train_protocol import training_lane
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.domain.shared.errors import DomainError, NotFoundError


@dataclass(frozen=True, kw_only=True)
class RetryRunCommand:
    run_id: uuid.UUID


class RetryRun:
    def __init__(self, runs: RunRepository, protocols: ProtocolRepository, enqueuer: JobEnqueuer, engines: EngineRegistry) -> None:
        self._runs = runs
        self._protocols = protocols
        self._enqueuer = enqueuer
        self._engines = engines

    async def __call__(self, command: RetryRunCommand, auth: AuthContext | None = None) -> Result[Run, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None
        run = await self._runs.get(auth.workspace_id, command.run_id)
        if run is None:
            return Failure(NotFoundError("Run", str(command.run_id)))
        try:
            lane = await self._lane(run, auth)
            run.retry()
        except DomainError as error:
            return Failure(error)
        # Update, then enqueue: a worker must never see a claimable row still reading FAILED.
        await self._runs.update(run)
        await self._enqueuer.enqueue(run.id, lane=lane)
        return Success(run)

    async def _lane(self, run: Run, auth: AuthContext) -> str:
        try:
            if run.kind is RunKind.TRAINING:
                return training_lane(self._engines, run.params["engine_id"], run.params.get("baseline_engine_id"))
            protocol = await self._protocols.get(auth.workspace_id, uuid.UUID(run.params["protocol_id"]))
            if protocol is None:
                raise NotFoundError("Protocol", run.params["protocol_id"])
            return self._engines.get(protocol.engine_id).manifest().lane
        except UnknownEngineError as error:
            raise NotFoundError("Engine", str(error)) from error
```

`container.py`: `container.define(RetryRun, lambda c: RetryRun(_runs(c), _protocols(c), c[JobEnqueuer], c[EngineRegistry]))`.

`routes/runs.py`:

```python
@router.post("/{run_id}/retry", status_code=204)
async def retry_run(run_id: uuid.UUID, auth: AuthDep, service: RetryRunDep) -> Response:
    result_to_response(await service(RetryRunCommand(run_id=run_id), auth=auth))
    return Response(status_code=204)
```

- [ ] **Step 3: `make generate-api`, tests, lint, commit**

```bash
git add -A && git commit -m "feat(runs): retry a failed or cancelled run in place

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 14: Runner management is an admin action

**Files:**
- Modify: `backend/src/daikonstudio/application/runners/manage.py:56-57,123-124` (`require_admin`)
- Modify: `backend/src/daikonstudio/interface/routes/runners.py:40-44` (`name` max length)
- Test: `backend/tests/unit/runners/test_manage.py`, `backend/tests/api/test_runners.py`

- [ ] **Step 1: Failing tests**

In `test_manage.py`: a fake auth with `workspace_role="editor"` → `CreateRunner` raises `AuthorizationError`; `"admin"` succeeds. In `test_runners.py`: the editor client (the api conftest's default client role; if it is `owner`, add an `editor_client` fixture mirroring `viewer_client` with role `editor`) gets 403 on `POST /api/v1/runners` and `POST /api/v1/runners/{id}/revoke`; `GET /api/v1/runners` stays 200 for a viewer. A 129-character name → 422.

- [ ] **Step 2: Implement**

`manage.py`: replace both `require_editor(auth)` with `require_admin(auth)`; update the module docstring: "an admin in some workspace: a runner token reaches every workspace's runs, so minting one is not an editor's call". `routes/runners.py`: `name: str = Field(max_length=128)`.

- [ ] **Step 3: Tests, lint, `make generate-api`, commit**

```bash
git commit -am "fix(runners): minting and revoking runner tokens requires the admin role

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 15: A confidence interval under the headline metric

**Files:**
- Modify: `backend/src/daikonstudio/application/execution/build_scorecard.py`
- Modify: `backend/src/daikonstudio/domain/execution/scorecard.py` (`primary_metric_ci`)
- Modify: `backend/src/daikonstudio/interface/routes/protocols.py` (`ScorecardResponse.primary_metric_ci`)
- Test: `backend/tests/unit/execution/test_scorecard.py`

**Interfaces:**
- Produces: `Scorecard.primary_metric_ci: tuple[float, float] | None`; `ScorecardResponse.primary_metric_ci: list[float] | None`; `primary_metric_ci(task, actual, predicted, *, resamples=1000, seed=0) -> tuple[float, float] | None`.

- [ ] **Step 1: Failing tests**

```python
from daikonstudio.application.execution.build_scorecard import primary_metric_ci


def test_the_regression_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    actual = list(rng.normal(size=200)); predicted = [a + e for a, e in zip(actual, rng.normal(scale=0.5, size=200))]
    rmse = float(np.sqrt(np.mean([(a - p) ** 2 for a, p in zip(actual, predicted)])))
    low, high = primary_metric_ci(TaskType.REGRESSION, actual, predicted)
    assert low < rmse < high
    assert high - low < 0.3


def test_the_classification_ci_is_over_mcc_at_the_half_threshold():
    actual = [1.0, 0.0] * 50
    predicted = [0.9, 0.1] * 40 + [0.1, 0.9] * 10
    low, high = primary_metric_ci(TaskType.BINARY_CLASSIFICATION, actual, predicted)
    assert 0.4 < low < 0.6 < high <= 1.0


def test_too_few_rows_means_no_interval_rather_than_a_misleading_one():
    assert primary_metric_ci(TaskType.REGRESSION, [1.0] * 10, [1.1] * 10) is None
```

(Also assert `regression_card().primary_metric_ci is not None` in the existing card helper test.)

- [ ] **Step 2: Implement**

```python
_CI_MIN_ROWS = 20
_CI_RESAMPLES = 1000


def _mcc(actual: np.ndarray, predicted_label: np.ndarray) -> float | None:
    tp = float(np.sum((actual == 1) & predicted_label)); tn = float(np.sum((actual == 0) & ~predicted_label))
    fp = float(np.sum((actual == 0) & predicted_label)); fn = float(np.sum((actual == 1) & ~predicted_label))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return None if denominator == 0 else (tp * tn - fp * fn) / denominator


def primary_metric_ci(task, actual, predicted, *, resamples=_CI_RESAMPLES, seed=0):
    """A 95 % bootstrap interval over the test set for the headline metric.

    Unpaired, and said so in the UI: the baseline's per-compound predictions are
    not persisted, so this is the sampling noise of *this* number, not a paired
    test of the difference. It is still what stops a +0.12 at n=197 reading as a
    win when the baseline sits inside [0.49, 0.76] (docs/roadmap.md, Traps)."""
    n = len(actual)
    if n < _CI_MIN_ROWS:
        return None
    a = np.asarray(actual, dtype=float); p = np.asarray(predicted, dtype=float)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        if task is TaskType.BINARY_CLASSIFICATION:
            value = _mcc(a[idx], p[idx] >= 0.5)
            if value is not None:
                values.append(value)
        else:
            values.append(float(np.sqrt(np.mean((a[idx] - p[idx]) ** 2))))
    if len(values) < resamples // 2:
        return None
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
```

`build_scorecard` passes `primary_metric_ci=primary_metric_ci(task, actual, predicted)` into `Scorecard`; `ScorecardResponse.primary_metric_ci = list(card.primary_metric_ci) if card.primary_metric_ci else None`. Imports: `import math`, `import numpy as np` (application already imports numpy in `build_profile.py`; update the module docstring sentence that says it holds no array dependency).

- [ ] **Step 3: `make generate-api`, tests, lint, commit**

```bash
git add -A && git commit -m "feat(scorecard): bootstrap 95% CI on the primary metric

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 16: Sessions that survive the hour

**Files:**
- Modify: `frontend/src/shared/lib/api/custom-instance.ts`
- Modify: `frontend/src/shared/lib/auth/config.ts` (expose the store)
- Create: `frontend/src/shared/lib/auth/session-renewal.ts`
- Modify: `frontend/src/app/(dashboard)/layout.tsx` (mount the watcher)
- Modify: `frontend/src/app/login/page.tsx` (show `?error=`)
- Test: `frontend/src/shared/lib/api/custom-instance.test.ts`, `frontend/src/shared/lib/auth/session-renewal.test.ts`

**Interfaces:**
- Produces: `setUnauthorizedHandler(handler: (() => void) | null)`; `customInstance` calls it once per 401 burst and throws `ApiError` with `silent: true`; `idpTokenExpiresAt(): number | null`; `useSessionRenewal()` hook that calls `getDuarClient().silentLogin()` when the IdP token is within 90 s of expiry and the tab is visible, or when a 401 arrives.

- [ ] **Step 1: Failing tests**

`custom-instance.test.ts`:

```ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, customInstance, setUnauthorizedHandler } from "./custom-instance";

vi.mock("@/shared/lib/auth/config", () => ({ getDuarClient: () => ({ isAuthenticated: false, getHeaders: () => ({}) }) }));

describe("401 handling", () => {
  afterEach(() => { setUnauthorizedHandler(null); vi.unstubAllGlobals(); });

  it("notifies the handler once and throws a silent ApiError", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "IdP token expired" }), { status: 401 })));
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    await expect(customInstance({ url: "/api/v1/runs", method: "GET" })).rejects.toMatchObject({ status: 401, silent: true });
    await expect(customInstance({ url: "/api/v1/runs", method: "GET" })).rejects.toBeInstanceOf(ApiError);
    expect(handler).toHaveBeenCalledTimes(1);
  });
});
```

`session-renewal.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { shouldRenew } from "./session-renewal";

describe("shouldRenew", () => {
  it("is false with no token or a token far from expiry", () => {
    expect(shouldRenew(null, 1_000_000)).toBe(false);
    expect(shouldRenew(1_000_000 + 3_600_000, 1_000_000)).toBe(false);
  });
  it("is true inside the 90 s window and after expiry", () => {
    expect(shouldRenew(1_000_000 + 60_000, 1_000_000)).toBe(true);
    expect(shouldRenew(1_000_000 - 1, 1_000_000)).toBe(true);
  });
});
```

- [ ] **Step 2: Implement**

`custom-instance.ts`: add

```ts
let _onUnauthorized: (() => void) | null = null;
let _unauthorizedNotified = false;

/** Registered by the dashboard layout; fired once per expiry, not once per failed query. */
export function setUnauthorizedHandler(handler: (() => void) | null): void {
  _onUnauthorized = handler;
  _unauthorizedNotified = false;
}
```

`ApiError` gains `readonly silent: boolean` (constructor param, default `false`). In the `!response.ok` branch, before throwing:

```ts
    if (response.status === 401 && _onUnauthorized) {
      if (!_unauthorizedNotified) {
        _unauthorizedNotified = true;
        _onUnauthorized();
      }
      throw new ApiError("Your session expired; signing you back in", 401, body, true);
    }
```

`config.ts`: keep the store in a module variable and export

```ts
let _store: AuthzLocalStorageStore | null = null;
export function idpTokenExpiresAt(): number | null {
  const token = _store?.getIdpToken();
  if (!token) return null;
  try {
    const payload = JSON.parse(atob(token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
    return typeof payload.exp === "number" ? payload.exp * 1000 : null;
  } catch {
    return null;
  }
}
```

`session-renewal.ts`:

```ts
"use client";
import { setUnauthorizedHandler } from "@/shared/lib/api/custom-instance";
import { getDuarClient, idpTokenExpiresAt } from "@/shared/lib/auth/config";
import { useEffect } from "react";

const RENEW_WINDOW_MS = 90_000;
const CHECK_EVERY_MS = 30_000;

/** Pure: renew when the IdP token is inside the window or already gone. */
export function shouldRenew(expiresAt: number | null, now: number): boolean {
  return expiresAt !== null && expiresAt - now < RENEW_WINDOW_MS;
}

/**
 * Google ID tokens live one hour and the implicit flow has no refresh token, so
 * the SDK can only renew by a prompt=none redirect through the IdP. Do that
 * ourselves shortly before expiry while the tab is visible, and on the first
 * 401 if we missed it. The redirect stores the current path and returns here.
 */
export function useSessionRenewal(): void {
  useEffect(() => {
    const renew = () => {
      if (document.visibilityState !== "visible") return;
      getDuarClient().silentLogin();
    };
    setUnauthorizedHandler(renew);
    const timer = window.setInterval(() => {
      if (shouldRenew(idpTokenExpiresAt(), Date.now())) renew();
    }, CHECK_EVERY_MS);
    return () => {
      window.clearInterval(timer);
      setUnauthorizedHandler(null);
    };
  }, []);
}
```

`layout.tsx`: call `useSessionRenewal()` inside `DashboardLayout`. `query-provider.tsx`: in `MutationCache.onError`, `if (error instanceof ApiError && error.silent) return;`.

`login/page.tsx`: read `useSearchParams().get("error")` and render it under the heading:

```tsx
{error && (
  <p className="mt-3 rounded-md border border-destructive/40 bg-destructive/5 p-2 text-xs text-destructive">{error}</p>
)}
```

(Wrap the page body in `<Suspense>` if Next complains about `useSearchParams` in a static page.)

- [ ] **Step 3: Gates, commit**

Run: `cd frontend && pnpm lint && pnpm exec tsc --noEmit && pnpm test`

```bash
git add -A && git commit -m "fix(auth): renew the IdP session before it expires and on 401; show login errors

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 17: No screen gets stuck

**Files:**
- Create: `frontend/src/app/error.tsx`, `frontend/src/app/not-found.tsx`
- Modify: `frontend/src/features/runs/components/run-detail.tsx`
- Modify: `frontend/src/features/runs/hooks/use-runs.ts` (stop polling on error)
- Modify: `frontend/src/features/protocols/hooks/use-protocols.ts` (`useRunPoll` same; publish invalidates on error)
- Modify: `frontend/src/features/sweeps/components/sweep-detail.tsx`, `frontend/src/features/sweeps/hooks/use-sweeps.ts`
- Modify: `frontend/src/features/protocols/components/train-protocol-form.tsx:144-152`
- Modify: `frontend/src/features/protocols/components/protocol-detail.tsx:116-117`
- Test: `frontend/src/features/runs/hooks/use-runs.test.ts`

**Interfaces:**
- Produces: `pollInterval(query) => number | false` in `use-runs.ts`, exported and reused by `useRunPoll` and `useSweep`: false when terminal **or** `query.state.status === "error"`.

- [ ] **Step 1: Failing test**

```ts
import { describe, expect, it } from "vitest";
import { pollInterval } from "./use-runs";

describe("pollInterval", () => {
  const q = (status: string, dataStatus?: string) => ({ state: { status, data: dataStatus ? { status: dataStatus } : undefined } }) as never;
  it("keeps polling a live run", () => expect(pollInterval(q("success", "running"))).toBe(2000));
  it("stops on a terminal status", () => expect(pollInterval(q("success", "failed"))).toBe(false));
  it("stops when the request itself fails", () => expect(pollInterval(q("error"))).toBe(false));
});
```

- [ ] **Step 2: Implement**

`use-runs.ts`:

```ts
export function pollInterval(query: { state: { status: string; data?: { status?: string } | undefined } }): number | false {
  if (query.state.status === "error") return false;
  return isTerminal(query.state.data?.status) ? false : RUN_POLL_MS;
}
```

used as `refetchInterval: pollInterval` in `useRun`, `useRunPoll`; `useSweep` adds the same error guard.

`run-detail.tsx`:
- destructure `isError, error` from `useRun`; after the skeleton branch add an error block ("Could not load this run" plus `error.message`) and return.
- Replace `{run.status === "ready" && protocol && (<TriageGrid …/>)}` with a kind switch: for `run.kind === "training"` render a card "This is a training run" linking to `/protocols/${run.protocol_id}` when set ("Its Scorecard is on the Protocol page") else "It produced no Protocol" with the error block above; the grid only for `run.kind === "prediction"`.
- Add the honest counts line when `run.metrics?.scored_rows != null`: `Scored {scored} of {uploaded} uploaded rows` plus `· {uploaded - scored} did not parse as structures` when they differ.
- Add a **Retry** button beside Cancel when `run.status === "failed" || run.status === "cancelled"`, via `useRetryRun()` (new hook mirroring `useCancelRun`, posting `/retry`, invalidating the run key, toasting "Run queued again").
- Add a pending hint when `run.status === "pending" && run.lane`: `useRunners()` from `@/features/runners`; if no runner with `status === "online"` includes `run.lane` in `lanes`, render `Waiting for a runner that serves the "{lane}" lane. None is online right now.`

`sweep-detail.tsx`: destructure `isError`; render the same error block instead of the skeleton when `isError`.

`train-protocol-form.tsx:144-152`: treat `cancelled` like `failed` with the toast "Training was cancelled"; on poll error (`run.isError`) toast once and `setRunId(undefined)`.

`use-protocols.ts::usePublishProtocol`: add `onError: (_e, id) => queryClient.invalidateQueries({ queryKey: [...PROTOCOL_KEY, id] })` so a 423 refreshes the stale "Draft".

`protocol-detail.tsx:116-117`: add `{scorecard.isError && <p className="text-sm text-destructive">Could not load the Scorecard.</p>}`.

`app/error.tsx`:

```tsx
"use client";
export default function Error({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <div className="flex min-h-screen items-center justify-center p-6">
      <div className="max-w-md space-y-3 text-center">
        <p className="text-lg font-semibold">Something went wrong</p>
        <p className="text-sm text-muted-foreground">{error.message}</p>
        <button type="button" className="rounded-md border px-3 py-1.5 text-sm" onClick={reset}>Try again</button>
      </div>
    </div>
  );
}
```

`app/not-found.tsx`: the same shape with "This page does not exist" and a link to `/`.

- [ ] **Step 3: Gates, commit**

```bash
git add -A && git commit -m "fix(ui): error boundaries, run and sweep error states, training-run detail, retry button, runner-lane hint

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 18: Lists that go past 50, pickers that see everything, one toast per failure

**Files:**
- Modify: `frontend/src/features/protocols/components/protocol-list.tsx`, `frontend/src/features/runs/components/run-list.tsx`, `frontend/src/features/collections/components/collection-list.tsx`, `frontend/src/features/sweeps/components/sweep-list.tsx`
- Modify: hooks `use-protocols.ts` (`useProtocols(cursor?, limit?)`), `use-runs.ts` (`useRuns(kind, cursor?)`), `use-collections.ts` (`useCollections(cursor?)`), `use-sweeps.ts` (`useSweeps(cursor?)` if the API paginates sweeps; else leave)
- Modify: pickers `train-protocol-form.tsx:77`, `sweep-form.tsx`, `predict-wizard.tsx:88` to request `limit: 200`
- Modify: `frontend/src/shared/providers/query-provider.tsx`, `frontend/src/features/datasets/components/dataset-wizard.tsx:139`, `use-datasets.ts::useCreateDataset`

**Interfaces:**
- Produces: a `LoadMore` pattern identical to `dataset-list.tsx:100-112` on the four lists; `useMutation` options may carry `meta: { silent: true }` and the global `MutationCache.onError` skips those.

- [ ] **Step 1: Implement the lists**

Copy the `cursor`/`pages`/`Load more` mechanism from `dataset-list.tsx` verbatim into each list; the hooks accept `cursor` and pass `params: { cursor }`. Pickers call `useDatasets(undefined, 200)` / `useProtocols(undefined, 200)` with `params: { limit }` (the server clamps to `MAX_PAGE_SIZE = 200`).

- [ ] **Step 2: One toast per failure**

`query-provider.tsx`:

```ts
          onError: (error, _variables, _context, mutation) => {
            if (mutation.meta?.silent) return;
            if (error instanceof ApiError && error.silent) return;
            showError(error instanceof Error ? error.message : "Operation failed");
          },
```

`useCreateDataset`: add `meta: { silent: true }` (the wizard renders the 422 report or its own toast). `usePublishProtocol`: `meta: { silent: true }` and toast only non-423 errors in its own `onError`. Remove the duplicate local `showError` in `dataset-wizard.tsx:139` only if the global one now fires for it; with `silent`, keep the local one.

- [ ] **Step 3: Gates, commit**

```bash
git add -A && git commit -m "fix(ui): load more on every list, full pickers, no duplicate failure toasts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 19: Identifiers in the wizard, grid and export; runner dialog truths

**Files:**
- Modify: `frontend/src/features/runs/components/predict-wizard.tsx`
- Modify: `frontend/src/features/runs/components/triage-grid.tsx`
- Modify: `frontend/src/features/runs/hooks/use-runs.ts` (`useCreateRun` accepts `id_column`)
- Modify: `frontend/src/features/runners/components/new-runner-dialog.tsx:40-50,183-186`
- Modify: `frontend/src/app/api/config/route.ts`, `frontend/src/shared/lib/app-config.tsx` (`runnerImage`)
- Test: `frontend/src/features/runs/lib/guess-id-column.test.ts`

**Interfaces:**
- Produces: `guessIdColumn(columns: string[], structureColumn: string): string | null` in `features/runs/lib/guess-id-column.ts`; `AppConfig.runnerImage: string` from `APP_RUNNER_IMAGE` (default `ghcr.io/sidxz/daikon-studio/api`), used by `runCommand(created, image)`.

- [ ] **Step 1: Failing test**

```ts
import { describe, expect, it } from "vitest";
import { guessIdColumn } from "./guess-id-column";

describe("guessIdColumn", () => {
  it("prefers an id-like header that is not the structure column", () => {
    expect(guessIdColumn(["Molecule Name", "smiles", "mw"], "smiles")).toBe("Molecule Name");
    expect(guessIdColumn(["compound_id", "smiles"], "smiles")).toBe("compound_id");
  });
  it("returns null when nothing looks like an identifier", () => {
    expect(guessIdColumn(["smiles", "mw"], "smiles")).toBe(null);
  });
});
```

- [ ] **Step 2: Implement**

```ts
const ID_PATTERN = /(^|[^a-z])(id|name|identifier|compound|cpd|sample|batch)([^a-z]|$)/i;
export function guessIdColumn(columns: string[], structureColumn: string): string | null {
  return columns.find((c) => c !== structureColumn && ID_PATTERN.test(c)) ?? null;
}
```

Wizard: `idColumn` state (default from `guessIdColumn` on drop, "none" allowed), a `Select` labelled "Identifier column (optional)" shown when `columns.length > 1`, passed as `id_column`. Grid: two columns before the readouts: `ID` (`field: "compound_id"`, hidden when every loaded row lacks it: track `hasIds` from the first block) and `Row` (`field: "input_row"`, width 80, header tooltip "Line in your uploaded file"). Both `sortable: false, filter: false`.

Runner dialog: `runCommand(created, image)` where the GPU variant uses `${image}-runner:gpu`? Keep it simple: `APP_RUNNER_IMAGE` is the CPU/API image name; the gpu command uses the same name with the tag `gpu` replaced by a second config `APP_RUNNER_GPU_IMAGE` defaulting to `daikon-runner:gpu`. Fix the copy: "STUDIO_URL above is the API's address as the runner machine must reach it."

- [ ] **Step 3: Gates, commit**

```bash
git add -A && git commit -m "feat(ui): optional identifier column through prediction and triage; configurable runner images

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 20: The verdict respects the interval

**Files:**
- Modify: `frontend/src/features/protocols/lib/verdict.ts`
- Modify: `frontend/src/features/protocols/components/scorecard-view.tsx:108-185`
- Test: `frontend/src/features/protocols/lib/verdict.test.ts`

**Interfaces:**
- Produces: `Verdict.ci?: [number, number] | null`; `computeVerdict` returns `kind: "within-noise"` with headline `"Ahead of the baseline, but within this test set's sampling noise"` when the baseline value lies inside `primary_metric_ci`.

- [ ] **Step 1: Failing tests**

```ts
  it("is within noise when the baseline sits inside the bootstrap interval", () => {
    const v = computeVerdict(scorecard({ metrics: { r2: 0.7 }, baseline_metrics: { r2: 0.6 }, primary_metric_ci: [0.5, 0.8] }));
    expect(v.kind).toBe("within-noise");
    expect(v.ci).toEqual([0.5, 0.8]);
  });
  it("still beats when the baseline is outside the interval", () => {
    expect(computeVerdict(scorecard({ primary_metric_ci: [0.65, 0.75] })).kind).toBe("beats");
  });
```

- [ ] **Step 2: Implement**

In `computeVerdict`, after the noise-floor branch:

```ts
  const ci = scorecard.primary_metric_ci ?? null;
  if (better && ci && baseline >= ci[0] && baseline <= ci[1]) {
    return { kind: "within-noise", headline: "Ahead of the baseline, but within this test set's sampling noise", model, baseline, delta, noiseFloor, ci };
  }
```

and carry `ci` on every returned verdict. In `VerdictBand`, under the numbers render when `ci`: `95% interval for this {metric}: [lo, hi] (bootstrap over the test set, unpaired)` with `ReadoutValue`s, and when `kind === "within-noise" && ci` the sentence: "The baseline's number sits inside that interval, so this test set cannot tell the two models apart."

- [ ] **Step 3: Gates, commit**

```bash
git add -A && git commit -m "feat(scorecard): the verdict reads the bootstrap interval

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 21: Contract, docs, roadmap, handoff

**Files:**
- Regenerate: `frontend/openapi.json`, `frontend/src/shared/lib/api/model/*`
- Rewrite: `backend/README.md`
- Modify: `docs/roadmap.md` (Traps: add the libgomp trap; mark per-lane timeouts shipped; note retry shipped; note the CI)
- Modify: `frontend/tests/e2e/api-mock.ts` (add `lane`, `metrics`, `input_row`, `compound_id` to mocked rows if the mock is typed against the generated models)

- [ ] **Step 1: `make generate-api` and confirm no diff beyond the tasks above**

Run: `make generate-api && git status --short frontend/openapi.json frontend/src/shared/lib/api/model | head`

- [ ] **Step 2: README**

Rewrite `backend/README.md` as the repo's developer README: what it is (two sentences), first-time setup (`cp backend/.env.example backend/.env`, `make install`, `make up`, `make dev`), ports table (8002 / 3003 / 5437), day-to-day targets, tests and gates, `make image-smoke`, the `uv add` footgun (re-run `make install` after any `uv add`, because a bare sync drops the extras), link to `deploy/README.md`, and the GPU image rebuild command for the human:

```bash
docker --context ned build -f backend/Dockerfile.gpu -t daikon-runner:gpu backend \
  && docker --context ned run --rm daikon-runner:gpu python -c "import chemprop, transformers, lightgbm; print('gpu image imports ok')"
```

Delete every sentence that says the frontend does not exist or that `make install` fails.

- [ ] **Step 3: Roadmap traps**

Add under **Traps**: "**A green `docker build` of the CPU image said nothing either.** LightGBM's wheel needs `libgomp.so.1`, absent from `python:3.13-slim`; the API image built clean for two months and crashed at boot. `make image-smoke` and CI now boot every image. Found 2026-10-02." Mark the per-lane timeout trap resolved with the setting name, and the retry plan as shipped.

- [ ] **Step 4: Full gates**

```bash
make lint && (cd backend && uv run lint-imports && env OMP_NUM_THREADS=1 uv run pytest tests/unit -q && uv run pytest tests/api tests/integration -q)
(cd frontend && pnpm lint && pnpm exec tsc --noEmit && pnpm test && pnpm build)
make image-smoke && make image-frontend
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "docs: README for the repo as it is; roadmap traps; regenerated contract

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 6: Whole-branch review**

Dispatch a fresh reviewer on the most capable model over `git diff main...beta-readiness` with the audit as the spec; fix Critical/Important findings in a final wave; then hand back with: the branch name, the commands to merge, the GPU rebuild command, and the two human steps left (register the beta domain in Google OAuth and Duar; create the first runner token in the UI).
