# daikon-studio — developer Makefile
#
# First run:
#   make install      # backend (uv) + frontend (pnpm) deps
#   make up           # start Postgres + Valkey, run DB migrations
#   make dev          # start backend (:8002) + frontend (:3003) + both job workers
#   open http://localhost:3003
#
# Two workers, because engines declare which lane they need and a worker serves one
# lane: the default lane runs the ECFP4 engines, the gpu lane runs chemprop (on CPU
# here). Set STUDIO_INLINE_JOBS=0 in backend/.env or neither is used.
#
# Day to day:  make logs (tail)  ·  make stop (stop servers)  ·  make down (stop containers)
#
# Auth uses the shared Sentinel identity-service (configured in backend/.env + frontend/.env.local).
# No local Sentinel is needed when SENTINEL_URL points at the remote service.

COMPOSE  := docker compose
BACKEND  := cd backend
FRONTEND := cd frontend
LOGDIR   := .logs
# Anchored to this Makefile's own directory, not the caller's CWD. `nuke` runs
# `rm -rf $(BLOBS)`, and a relative path there would resolve against wherever make
# was invoked from -- `make -f /path/to/daikon-studio/Makefile nuke` run from / would
# aim it at /.blobs. An absolute path derived from the Makefile cannot drift.
ROOT     := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
BLOBS    := $(ROOT)/.blobs
# Load backend/.env (DATABASE_URL, SENTINEL_*) into the recipe shell.
BE_ENV   := set -a && . ./.env && set +a
# arq worker entrypoint (runs the training and prediction jobs the API enqueues).
ARQ      := uv run arq daikonstudio.infrastructure.worker.WorkerSettings
# OMP_NUM_THREADS=1 is load-bearing, not tuning. torch and scikit-learn each ship
# their own libomp.dylib, and a training job loads BOTH -- RunTraining fits the
# chosen engine and the mandatory ECFP4 baseline in one process, by design. Three
# OpenMP runtimes in one process is undefined behaviour and it segfaults partway
# through a real chemprop fit (EXC_BAD_ACCESS in __kmp_fork_barrier, reproduced
# against BBBP). Pinning OpenMP to one thread removes the thread teams they fight
# over. KMP_DUPLICATE_LIB_OK does NOT fix it -- it was already set when this crashed.
# ponytail: costs some intra-op parallelism in the ECFP4 baseline's fit. Revisit only
# with a measurement; a slower baseline beats a worker that dies mid-run.
WORKER     := env OMP_NUM_THREADS=1 $(ARQ)
# The same entrypoint bound to the `gpu` lane. Engines declare a lane on their
# manifest (chemprop-dmpnn declares "gpu"); a worker serves exactly one lane, so
# without this process a chemprop run sits PENDING forever with nothing to pull it.
# Locally there is no GPU and chemprop falls back to CPU -- slow, but it is the same
# code path the real GPU worker runs, so the dev loop exercises lane routing,
# background execution, per-epoch progress and cancellation for real.
# MAX_JOBS=1 mirrors production, where concurrent fits would exhaust device memory.
WORKER_GPU := env OMP_NUM_THREADS=1 STUDIO_WORKER_LANE=gpu STUDIO_WORKER_MAX_JOBS=1 $(ARQ)

.DEFAULT_GOAL := help
.PHONY: help up down install dev dev-be dev-fe dev-worker dev-worker-gpu stop logs migrate \
        generate-api test test-api test-all test-fe lint lint-fe nuke

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

up: ## Start Postgres + Valkey, wait for readiness, run migrations
	$(COMPOSE) up -d postgres valkey
	@echo "Waiting for Postgres on :5435..."
	@until $(COMPOSE) exec -T postgres pg_isready -U studio -q 2>/dev/null; do sleep 1; done
	@$(MAKE) --no-print-directory migrate
	@echo "Infra ready: Postgres :5435, Valkey :6381."

down: ## Stop containers (keep data)
	$(COMPOSE) stop

install: ## Install backend (uv) + frontend (pnpm) dependencies
	$(BACKEND) && uv sync
	$(FRONTEND) && pnpm install

migrate: ## Apply DB migrations (alembic)
	$(BACKEND) && $(BE_ENV) && uv run alembic upgrade head

dev: stop ## Start backend (:8002) + frontend (:3003) + job worker in the background
	@mkdir -p $(LOGDIR)
	@echo "Starting backend on :8002..."
	@nohup sh -c '$(BACKEND) && $(BE_ENV) && exec uv run uvicorn daikonstudio.interface.app:app --reload --port 8002' \
		> $(LOGDIR)/backend.log 2>&1 & echo "$$!" > $(LOGDIR)/backend.pid
	@echo "Starting frontend on :3003..."
	@nohup sh -c '$(FRONTEND) && exec pnpm dev' \
		> $(LOGDIR)/frontend.log 2>&1 & echo "$$!" > $(LOGDIR)/frontend.pid
	@echo "Starting job worker (default lane)..."
	@nohup sh -c '$(BACKEND) && $(BE_ENV) && exec $(WORKER)' \
		> $(LOGDIR)/worker.log 2>&1 & echo "$$!" > $(LOGDIR)/worker.pid
	@echo "Starting job worker (gpu lane)..."
	@nohup sh -c '$(BACKEND) && $(BE_ENV) && exec $(WORKER_GPU)' \
		> $(LOGDIR)/worker-gpu.log 2>&1 & echo "$$!" > $(LOGDIR)/worker-gpu.pid
	@sleep 1
	@echo ""
	@echo "  Backend   http://localhost:8002/docs   (pid $$(cat $(LOGDIR)/backend.pid), log $(LOGDIR)/backend.log)"
	@echo "  Frontend  http://localhost:3003        (pid $$(cat $(LOGDIR)/frontend.pid), log $(LOGDIR)/frontend.log)"
	@echo "  Worker    default lane: ecfp4 engines                  (pid $$(cat $(LOGDIR)/worker.pid), log $(LOGDIR)/worker.log)"
	@echo "  Worker    gpu lane: chemprop (on CPU locally)          (pid $$(cat $(LOGDIR)/worker-gpu.pid), log $(LOGDIR)/worker-gpu.log)"
	@echo "  make logs — tail all    ·    make stop — stop all"
	@echo ""
	@echo "  NOTE: both workers only matter when STUDIO_INLINE_JOBS=0 in backend/.env."
	@echo "        With inline jobs the API runs the fit inside the HTTP request, which"
	@echo "        for chemprop means the browser hangs for minutes and times out."

dev-be: ## (Re)start the backend only, in the background
	@mkdir -p $(LOGDIR)
	@lsof -ti:8002 | xargs kill 2>/dev/null || true
	@nohup sh -c '$(BACKEND) && $(BE_ENV) && exec uv run uvicorn daikonstudio.interface.app:app --reload --port 8002' \
		> $(LOGDIR)/backend.log 2>&1 & echo "$$!" > $(LOGDIR)/backend.pid
	@echo "Backend (re)started on :8002 (log $(LOGDIR)/backend.log)"

dev-fe: ## (Re)start the frontend only, in the background
	@mkdir -p $(LOGDIR)
	@lsof -ti:3003 | xargs kill 2>/dev/null || true
	@nohup sh -c '$(FRONTEND) && exec pnpm dev' \
		> $(LOGDIR)/frontend.log 2>&1 & echo "$$!" > $(LOGDIR)/frontend.pid
	@echo "Frontend (re)started on :3003 (log $(LOGDIR)/frontend.log)"

# Both worker targets kill by PID file only, deliberately -- no pkill. The two
# workers differ solely by the STUDIO_WORKER_LANE in their environment, so their
# command lines are identical and `pkill -f 'arq ...WorkerSettings'` cannot tell
# them apart: restarting one would silently kill the other. `make stop` still
# pkills, because there killing every worker is the intent.
dev-worker: ## (Re)start the default-lane worker only, in the background
	@mkdir -p $(LOGDIR)
	@[ -f $(LOGDIR)/worker.pid ] && kill $$(cat $(LOGDIR)/worker.pid) 2>/dev/null || true
	@nohup sh -c '$(BACKEND) && $(BE_ENV) && exec $(WORKER)' \
		> $(LOGDIR)/worker.log 2>&1 & echo "$$!" > $(LOGDIR)/worker.pid
	@echo "Default-lane worker (re)started (log $(LOGDIR)/worker.log)"

dev-worker-gpu: ## (Re)start the gpu-lane worker only (chemprop; CPU locally)
	@mkdir -p $(LOGDIR)
	@[ -f $(LOGDIR)/worker-gpu.pid ] && kill $$(cat $(LOGDIR)/worker-gpu.pid) 2>/dev/null || true
	@nohup sh -c '$(BACKEND) && $(BE_ENV) && exec $(WORKER_GPU)' \
		> $(LOGDIR)/worker-gpu.log 2>&1 & echo "$$!" > $(LOGDIR)/worker-gpu.pid
	@echo "GPU-lane worker (re)started (log $(LOGDIR)/worker-gpu.log)"

stop: ## Stop the backend + frontend + worker dev processes
	@[ -f $(LOGDIR)/backend.pid ]  && kill $$(cat $(LOGDIR)/backend.pid)  2>/dev/null || true
	@[ -f $(LOGDIR)/frontend.pid ] && kill $$(cat $(LOGDIR)/frontend.pid) 2>/dev/null || true
	@[ -f $(LOGDIR)/worker.pid ]   && kill $$(cat $(LOGDIR)/worker.pid)   2>/dev/null || true
	@[ -f $(LOGDIR)/worker-gpu.pid ] && kill $$(cat $(LOGDIR)/worker-gpu.pid) 2>/dev/null || true
	@lsof -ti:8002 | xargs kill 2>/dev/null || true
	@lsof -ti:3003 | xargs kill 2>/dev/null || true
	@pkill -f 'arq daikonstudio.infrastructure.worker.WorkerSettings' 2>/dev/null || true
	@rm -f $(LOGDIR)/backend.pid $(LOGDIR)/frontend.pid $(LOGDIR)/worker.pid $(LOGDIR)/worker-gpu.pid
	@echo "Dev servers stopped."

logs: ## Tail backend + frontend + both worker dev logs
	@mkdir -p $(LOGDIR) && touch $(LOGDIR)/backend.log $(LOGDIR)/frontend.log \
		$(LOGDIR)/worker.log $(LOGDIR)/worker-gpu.log
	tail -f $(LOGDIR)/backend.log $(LOGDIR)/frontend.log $(LOGDIR)/worker.log $(LOGDIR)/worker-gpu.log

generate-api: ## Refresh the OpenAPI snapshot from the backend + regenerate the TS client
	$(BACKEND) && $(BE_ENV) && uv run python -c \
		"import json,sys; from daikonstudio.interface.app import app; sys.stdout.write(json.dumps(app.openapi(), indent=2))" \
		> ../frontend/openapi.json
	$(FRONTEND) && pnpm generate:api

test: ## Backend unit tests + import-linter
	$(BACKEND) && env OMP_NUM_THREADS=1 uv run pytest tests/unit -v && uv run lint-imports

test-api: ## Backend API tests
	$(BACKEND) && uv run pytest tests/api -v

test-all: ## All backend tests + import-linter
	$(BACKEND) && env OMP_NUM_THREADS=1 uv run pytest -v && uv run lint-imports

test-fe: ## Frontend tests (vitest)
	$(FRONTEND) && pnpm test

lint: ## Backend lint (ruff + mypy)
	$(BACKEND) && uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src

lint-fe: ## Frontend lint (biome)
	$(FRONTEND) && pnpm lint

nuke: ## Stop containers and DELETE all data volumes + local blobs
	$(COMPOSE) down -v
	# Blobs go too, deliberately. A dropped database with the blob store left
	# behind is the worse of the two inconsistent states: orphaned snapshots and
	# artifacts nothing references, under ids the fresh database will never mint.
	rm -rf $(BLOBS)
	@echo "Nuked: database volume and $(BLOBS). Run 'make up' to recreate."
