# daikon-studio — developer Makefile
#
# First run:
#   make install      # backend (uv) + frontend (pnpm) deps
#   make up           # start Postgres, run DB migrations, seed the dev runners
#   make dev          # start backend (:8002) + frontend (:3003) + both runner agents
#   open http://localhost:3003
#
# Two runner agents, because engines declare which lane they need and a runner serves
# the lanes on its own row: the default lane runs the ECFP4 engines, the gpu lane runs
# chemprop (on the Mac GPU via MPS here). Set STUDIO_INLINE_JOBS=0 in backend/.env or
# neither is used.
#
# Day to day:  make logs (tail)  ·  make stop (stop servers)  ·  make down (stop containers)
#
# Auth uses the shared Duar identity-service (configured in backend/.env + frontend/.env.local).
# No local Duar is needed when DUAR_URL points at the remote service.

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
# Load backend/.env (DATABASE_URL, DUAR_*) into the recipe shell.
BE_ENV   := set -a && . ./.env && set +a
# Export APP_VERSION / APP_GIT_SHA / APP_BUILD_DATE for a component, from the
# same script CI bakes into the images (RELEASING.md). $(call BUILD_INFO,backend)
BUILD_INFO = eval "$$($(ROOT)/scripts/build-info.sh $(1) | sed s/^/export\ APP_/)"
# Runner agents replace the arq workers: same jobs, but claimed over the HTTP
# runner protocol (see backend/README.md).
# Lanes live on the server-side runner rows that `make seed-runners` ensures.
# OMP_NUM_THREADS=1 is load-bearing -- see the original explanation below (kept).
RUNNER     := env OMP_NUM_THREADS=1 STUDIO_URL=http://localhost:8002 uv run python -m daikonstudio.infrastructure.runner
# torch and scikit-learn each ship their own libomp.dylib, and a training job loads
# BOTH -- RunTraining fits the chosen engine and the mandatory ECFP4 baseline in one
# process, by design. Three OpenMP runtimes in one process is undefined behaviour and
# it segfaults partway through a real chemprop fit (EXC_BAD_ACCESS in
# __kmp_fork_barrier, reproduced against BBBP). Pinning OpenMP to one thread removes
# the thread teams they fight over. KMP_DUPLICATE_LIB_OK does NOT fix it -- it was
# already set when this crashed.
# ponytail: costs some intra-op parallelism in the ECFP4 baseline's fit. Revisit only
# with a measurement; a slower baseline beats an agent that dies mid-run.
WORKER     := env STUDIO_RUNNER_TOKEN=drt_dev_default $(RUNNER)
# The same entrypoint carrying the `dev-local-gpu` runner's token, which registered
# only the `gpu` lane (see `infrastructure/runner/seed.py`). Engines declare a lane on
# their manifest (chemprop-dmpnn declares "gpu"); a runner only claims the lanes on
# its own row, so without this process a chemprop run sits PENDING forever with
# nothing to claim it.
#
# On Apple Silicon this really is GPU-backed: lightning's `accelerator="auto"`
# resolves to MPSAccelerator (verified -- a full chemprop fit and predict both run on
# `mps:0` with no `PYTORCH_ENABLE_MPS_FALLBACK` and no unimplemented-op errors). It is
# NOT a container: Docker Desktop runs a Linux VM that cannot see Metal, so there is
# no Mac equivalent of `--gpus all` and no Mac GPU image to build. The Mac GPU path is
# this native process; `Dockerfile.gpu` is the CUDA path and is a different machine.
#
# The run's phase names the device it resolved to, so a runner registered for the gpu
# lane that is quietly on CPU says so rather than just being slow.
WORKER_GPU := env STUDIO_RUNNER_TOKEN=drt_dev_gpu $(RUNNER)

.DEFAULT_GOAL := help
.PHONY: help up down install dev dev-be dev-fe dev-worker dev-worker-gpu stop logs migrate \
        seed-runners backfill-maps register-protocols generate-api test test-api test-all test-fe lint lint-fe nuke \
        image-runner-cpu image-runner-gpu image-smoke image-frontend security-scan publish-runner-gpu

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

up: ## Start Postgres, wait for readiness, run migrations, seed the dev runners
	$(COMPOSE) up -d postgres
	@echo "Waiting for Postgres on :5437..."
	@until $(COMPOSE) exec -T postgres pg_isready -U studio -q 2>/dev/null; do sleep 1; done
	@$(MAKE) --no-print-directory migrate
	@$(MAKE) --no-print-directory seed-runners
	@echo "Infra ready: Postgres :5437."

down: ## Stop containers (keep data)
	$(COMPOSE) stop

install: ## Install backend (uv) + frontend (pnpm) dependencies
	$(BACKEND) && uv sync --extra gpu --extra s3
	$(FRONTEND) && pnpm install

migrate: ## Apply DB migrations (alembic)
	$(BACKEND) && $(BE_ENV) && uv run alembic upgrade head

backfill-maps: ## Build chemical-space maps (and run neighbours) for protocols that lack one
	$(BACKEND) && $(BE_ENV) && uv run python -m daikonstudio.infrastructure.backfill_maps

register-protocols: ## Register existing protocols with Duar (drafts private, published workspace-wide)
	$(BACKEND) && $(BE_ENV) && uv run python -m daikonstudio.infrastructure.duar.register_protocols

seed-runners: ## Ensure the two local dev runners exist
	$(BACKEND) && $(BE_ENV) && STUDIO_DEV_SEED=1 uv run python -m daikonstudio.infrastructure.runner.seed

dev: stop ## Start backend (:8002) + frontend (:3003) + both runner agents in the background
	@mkdir -p $(LOGDIR)
	@echo "Starting backend on :8002..."
	@nohup sh -c '$(call BUILD_INFO,backend) && $(BACKEND) && $(BE_ENV) && exec uv run uvicorn daikonstudio.interface.app:app --reload --port 8002' \
		> $(LOGDIR)/backend.log 2>&1 & echo "$$!" > $(LOGDIR)/backend.pid
	@echo "Starting frontend on :3003..."
	@nohup sh -c '$(call BUILD_INFO,frontend) && $(FRONTEND) && exec pnpm dev' \
		> $(LOGDIR)/frontend.log 2>&1 & echo "$$!" > $(LOGDIR)/frontend.pid
	@echo "Starting runner agent (default lane)..."
	@nohup sh -c '$(BACKEND) && $(BE_ENV) && exec $(WORKER)' \
		> $(LOGDIR)/worker.log 2>&1 & echo "$$!" > $(LOGDIR)/worker.pid
	@echo "Starting runner agent (gpu lane)..."
	@nohup sh -c '$(BACKEND) && $(BE_ENV) && exec $(WORKER_GPU)' \
		> $(LOGDIR)/worker-gpu.log 2>&1 & echo "$$!" > $(LOGDIR)/worker-gpu.pid
	@sleep 1
	@echo ""
	@echo "  Backend   http://localhost:8002/docs   (pid $$(cat $(LOGDIR)/backend.pid), log $(LOGDIR)/backend.log)"
	@echo "  Frontend  http://localhost:3003        (pid $$(cat $(LOGDIR)/frontend.pid), log $(LOGDIR)/frontend.log)"
	@echo "  Runner    default lane: ecfp4 engines                  (pid $$(cat $(LOGDIR)/worker.pid), log $(LOGDIR)/worker.log)"
	@echo "  Runner    gpu lane: chemprop (Mac GPU via MPS)         (pid $$(cat $(LOGDIR)/worker-gpu.pid), log $(LOGDIR)/worker-gpu.log)"
	@echo "  make logs — tail all    ·    make stop — stop all"
	@echo ""
	@echo "  NOTE: both runner agents only matter when STUDIO_INLINE_JOBS=0 in backend/.env."
	@echo "        With inline jobs the API runs the fit inside the HTTP request, which"
	@echo "        for chemprop means the browser hangs for minutes and times out."

dev-be: ## (Re)start the backend only, in the background
	@mkdir -p $(LOGDIR)
	@lsof -ti tcp:8002 -sTCP:LISTEN | xargs kill 2>/dev/null || true
	@# Wait for the old server to release the port, or the new one dies with EADDRINUSE.
	@for i in $$(seq 50); do lsof -ti tcp:8002 -sTCP:LISTEN >/dev/null || break; sleep 0.2; done
	@nohup sh -c '$(call BUILD_INFO,backend) && $(BACKEND) && $(BE_ENV) && exec uv run uvicorn daikonstudio.interface.app:app --reload --port 8002' \
		> $(LOGDIR)/backend.log 2>&1 & echo "$$!" > $(LOGDIR)/backend.pid
	@echo "Backend (re)started on :8002 (log $(LOGDIR)/backend.log)"

dev-fe: ## (Re)start the frontend only, in the background
	@mkdir -p $(LOGDIR)
	@lsof -ti tcp:3003 -sTCP:LISTEN | xargs kill 2>/dev/null || true
	@# Wait for the old server to release the port, or the new one dies with EADDRINUSE.
	@for i in $$(seq 50); do lsof -ti tcp:3003 -sTCP:LISTEN >/dev/null || break; sleep 0.2; done
	@nohup sh -c '$(call BUILD_INFO,frontend) && $(FRONTEND) && exec pnpm dev' \
		> $(LOGDIR)/frontend.log 2>&1 & echo "$$!" > $(LOGDIR)/frontend.pid
	@echo "Frontend (re)started on :3003 (log $(LOGDIR)/frontend.log)"

# Both runner-agent targets kill by PID file only, deliberately -- no pkill. The two
# agents differ solely by the STUDIO_RUNNER_TOKEN in their environment, so their
# command lines are identical and `pkill -f 'daikonstudio.infrastructure.runner'`
# cannot tell them apart: restarting one would silently kill the other. `make stop`
# still pkills, because there killing every agent is the intent.
dev-worker: ## (Re)start the default-lane runner agent only, in the background
	@mkdir -p $(LOGDIR)
	@[ -f $(LOGDIR)/worker.pid ] && kill $$(cat $(LOGDIR)/worker.pid) 2>/dev/null || true
	@nohup sh -c '$(BACKEND) && $(BE_ENV) && exec $(WORKER)' \
		> $(LOGDIR)/worker.log 2>&1 & echo "$$!" > $(LOGDIR)/worker.pid
	@echo "Default-lane runner agent (re)started (log $(LOGDIR)/worker.log)"

dev-worker-gpu: ## (Re)start the gpu-lane runner agent only (chemprop; Mac GPU via MPS)
	@mkdir -p $(LOGDIR)
	@[ -f $(LOGDIR)/worker-gpu.pid ] && kill $$(cat $(LOGDIR)/worker-gpu.pid) 2>/dev/null || true
	@nohup sh -c '$(BACKEND) && $(BE_ENV) && exec $(WORKER_GPU)' \
		> $(LOGDIR)/worker-gpu.log 2>&1 & echo "$$!" > $(LOGDIR)/worker-gpu.pid
	@echo "GPU-lane runner agent (re)started (log $(LOGDIR)/worker-gpu.log)"

stop: ## Stop the backend + frontend + runner-agent dev processes
	@[ -f $(LOGDIR)/backend.pid ]  && kill $$(cat $(LOGDIR)/backend.pid)  2>/dev/null || true
	@[ -f $(LOGDIR)/frontend.pid ] && kill $$(cat $(LOGDIR)/frontend.pid) 2>/dev/null || true
	@[ -f $(LOGDIR)/worker.pid ]   && kill $$(cat $(LOGDIR)/worker.pid)   2>/dev/null || true
	@[ -f $(LOGDIR)/worker-gpu.pid ] && kill $$(cat $(LOGDIR)/worker-gpu.pid) 2>/dev/null || true
	@lsof -ti tcp:8002 -sTCP:LISTEN | xargs kill 2>/dev/null || true
	@lsof -ti tcp:3003 -sTCP:LISTEN | xargs kill 2>/dev/null || true
	@pkill -f 'daikonstudio.infrastructure.runner' 2>/dev/null || true
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

image-runner-cpu: ## Build the API + default-lane runner image locally as daikon-runner:cpu (the Runners UI names APP_RUNNER_IMAGE, the published one, by default)
	$(call BUILD_INFO,backend) && docker build -f backend/Dockerfile -t daikon-runner:cpu \
		--build-arg APP_VERSION --build-arg APP_GIT_SHA --build-arg APP_BUILD_DATE backend

# The CUDA runner image is built, GPU-tested, scanned and pushed on atlantic: x86_64, an
# NVIDIA GPU, a lasting layer cache, and room for a ~22 GB image a GitHub-hosted runner
# lacks. Driven over the `atlantic` docker context (SSH) rather than registered as a
# self-hosted GitHub runner, because this repo is public and any fork PR could target one.
GPU_CONTEXT ?= atlantic
GPU_IMAGE   := ghcr.io/sidxz/daikon-studio/runner-gpu
GPU_DOCKER  := docker --context $(GPU_CONTEXT)
# Pushes run on atlantic's own docker CLI: over an SSH context the CLI, and so the registry
# login it sends, is this machine's, and the ghcr login lives on atlantic (RELEASING.md).
GPU_SSH      = ssh $(patsubst ssh://%,%,$(shell docker context inspect $(GPU_CONTEXT) --format '{{.Endpoints.docker.Host}}'))

image-runner-gpu: ## Build the CUDA runner image on atlantic as daikon-runner:gpu (backend/Dockerfile.gpu)
	$(call BUILD_INFO,backend) && $(GPU_DOCKER) build -f backend/Dockerfile.gpu -t daikon-runner:gpu \
		--build-arg APP_VERSION --build-arg APP_GIT_SHA --build-arg APP_BUILD_DATE backend

# ponytail: the scan runs Trivy in a container on atlantic, which cannot see a local
# .trivyignore (there is none yet); mount or copy it in once one exists.
publish-runner-gpu: ## Backend release: build the CUDA runner on atlantic, test it on the GPU, Trivy-scan it, push to ghcr
	@git describe --exact-match --tags --match 'backend-v*' HEAD >/dev/null 2>&1 && git diff --quiet HEAD \
	  || { echo "publish-runner-gpu: check out a backend-v* tag, with a clean tree, first (RELEASING.md)"; exit 1; }
	$(MAKE) image-runner-gpu
	$(GPU_DOCKER) run --rm --gpus all daikon-runner:gpu python -c "import torch, chemprop; \
		assert torch.cuda.is_available(), 'torch cannot see the GPU: host driver too old for this torch build?'; \
		x = torch.ones(1024, 1024, device='cuda'); print(torch.__version__, torch.cuda.get_device_name(0), float((x @ x).sum()))"
	$(GPU_DOCKER) run --rm -v /var/run/docker.sock:/var/run/docker.sock -v daikon-trivy-cache:/root/.cache \
		aquasec/trivy:0.75.0 image --scanners vuln --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 daikon-runner:gpu
	$(call BUILD_INFO,backend) && case "$$APP_VERSION" in *-*) tags=$$APP_VERSION ;; \
		*) tags="$$APP_VERSION $${APP_VERSION%.*} $${APP_VERSION%%.*} latest" ;; esac \
	  && for t in $$tags; do $(GPU_DOCKER) tag daikon-runner:gpu $(GPU_IMAGE):$$t && $(GPU_SSH) docker push $(GPU_IMAGE):$$t || exit 1; done

# A green build says nothing about whether the image works (docs/roadmap.md, Traps):
# the CPU image built clean for two months while LightGBM could not import inside it.
# This imports the app and every engine, so a missing shared library fails here.
image-smoke: image-runner-cpu ## Build the CPU image and prove it can import the app and every engine
	docker run --rm -e STUDIO_DUAR_SERVICE_KEY=smoke -e STUDIO_IDP_AUDIENCE=smoke daikon-runner:cpu \
		python -c "import daikonstudio.interface.app, daikonstudio.infrastructure.engines.registry as r; print('engines:', sorted(m.id for m in r.default_registry().manifests()))"

image-frontend: ## Build the daikon-frontend:local image (Next standalone + RDKit wasm)
	$(call BUILD_INFO,frontend) && docker build -f frontend/Dockerfile -t daikon-frontend:local \
		--build-arg APP_VERSION --build-arg APP_GIT_SHA --build-arg APP_BUILD_DATE frontend

security-scan: image-runner-cpu image-frontend ## Trivy gate before a release: lockfiles + secrets, then both images; fails on fixable HIGH/CRITICAL
	@command -v trivy >/dev/null || { echo "trivy is not installed: brew install trivy"; exit 1; }
	trivy fs --scanners vuln,secret --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 \
	  --skip-dirs node_modules,.venv,.next,.claude,.logs,.blobs,.superpowers,docs .
	trivy image --scanners vuln --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 daikon-runner:cpu
	trivy image --scanners vuln --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 daikon-frontend:local

nuke: ## Stop containers and DELETE all data volumes + local blobs
	$(COMPOSE) down -v
	# Blobs go too, deliberately. A dropped database with the blob store left
	# behind is the worse of the two inconsistent states: orphaned snapshots and
	# artifacts nothing references, under ids the fresh database will never mint.
	rm -rf $(BLOBS)
	@echo "Nuked: database volume and $(BLOBS). Run 'make up' to recreate."
