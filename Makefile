# daikon-studio — developer Makefile
#
# First run:
#   make install      # backend (uv) + frontend (pnpm) deps
#   make up           # start Postgres + Valkey, run DB migrations
#   make dev          # start backend (:8002) + frontend (:3002) + import worker in the background
#   open http://localhost:3002
#
# Day to day:  make logs (tail)  ·  make stop (stop servers)  ·  make down (stop containers)
#
# Auth uses the shared Sentinel identity-service (configured in backend/.env + frontend/.env.local).
# No local Sentinel is needed when SENTINEL_URL points at the remote service.

COMPOSE  := docker compose
BACKEND  := cd backend
FRONTEND := cd frontend
LOGDIR   := .logs
# Load backend/.env (DATABASE_URL, SENTINEL_*) into the recipe shell.
BE_ENV   := set -a && . ./.env && set +a
# arq import worker entrypoint (processes FE-enqueued import jobs).
WORKER   := uv run arq daikonstudio.infrastructure.ingestion.worker.WorkerSettings

.DEFAULT_GOAL := help
.PHONY: help up down install dev dev-be dev-fe dev-worker stop logs migrate generate-api \
        test test-api test-all test-fe lint lint-fe nuke

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

up: ## Start Postgres + Valkey, wait for readiness, run migrations
	$(COMPOSE) up -d postgres valkey
	@echo "Waiting for Postgres on :5434..."
	@until $(COMPOSE) exec -T postgres pg_isready -U daikonstudio -q 2>/dev/null; do sleep 1; done
	@$(MAKE) --no-print-directory migrate
	@echo "Infra ready: Postgres :5434, Valkey :6381."

down: ## Stop containers (keep data)
	$(COMPOSE) stop

install: ## Install backend (uv) + frontend (pnpm) dependencies
	$(BACKEND) && uv sync
	$(FRONTEND) && pnpm install

migrate: ## Apply DB migrations (alembic)
	$(BACKEND) && $(BE_ENV) && uv run alembic upgrade head

dev: stop ## Start backend (:8002) + frontend (:3002) + import worker in the background
	@mkdir -p $(LOGDIR)
	@echo "Starting backend on :8002..."
	@nohup sh -c '$(BACKEND) && $(BE_ENV) && exec uv run uvicorn daikonstudio.interface.app:app --reload --port 8002' \
		> $(LOGDIR)/backend.log 2>&1 & echo "$$!" > $(LOGDIR)/backend.pid
	@echo "Starting frontend on :3002..."
	@nohup sh -c '$(FRONTEND) && exec pnpm dev' \
		> $(LOGDIR)/frontend.log 2>&1 & echo "$$!" > $(LOGDIR)/frontend.pid
	@echo "Starting import worker..."
	@nohup sh -c '$(BACKEND) && $(BE_ENV) && exec $(WORKER)' \
		> $(LOGDIR)/worker.log 2>&1 & echo "$$!" > $(LOGDIR)/worker.pid
	@sleep 1
	@echo ""
	@echo "  Backend   http://localhost:8002/docs   (pid $$(cat $(LOGDIR)/backend.pid), log $(LOGDIR)/backend.log)"
	@echo "  Frontend  http://localhost:3002        (pid $$(cat $(LOGDIR)/frontend.pid), log $(LOGDIR)/frontend.log)"
	@echo "  Worker    import jobs                   (pid $$(cat $(LOGDIR)/worker.pid), log $(LOGDIR)/worker.log)"
	@echo "  make logs — tail all    ·    make stop — stop all"

dev-be: ## (Re)start the backend only, in the background
	@mkdir -p $(LOGDIR)
	@lsof -ti:8002 | xargs kill 2>/dev/null || true
	@nohup sh -c '$(BACKEND) && $(BE_ENV) && exec uv run uvicorn daikonstudio.interface.app:app --reload --port 8002' \
		> $(LOGDIR)/backend.log 2>&1 & echo "$$!" > $(LOGDIR)/backend.pid
	@echo "Backend (re)started on :8002 (log $(LOGDIR)/backend.log)"

dev-fe: ## (Re)start the frontend only, in the background
	@mkdir -p $(LOGDIR)
	@lsof -ti:3002 | xargs kill 2>/dev/null || true
	@nohup sh -c '$(FRONTEND) && exec pnpm dev' \
		> $(LOGDIR)/frontend.log 2>&1 & echo "$$!" > $(LOGDIR)/frontend.pid
	@echo "Frontend (re)started on :3002 (log $(LOGDIR)/frontend.log)"

dev-worker: ## (Re)start the import worker only, in the background
	@mkdir -p $(LOGDIR)
	@[ -f $(LOGDIR)/worker.pid ] && kill $$(cat $(LOGDIR)/worker.pid) 2>/dev/null || true
	@pkill -f 'arq daikonstudio.infrastructure.ingestion.worker.WorkerSettings' 2>/dev/null || true
	@nohup sh -c '$(BACKEND) && $(BE_ENV) && exec $(WORKER)' \
		> $(LOGDIR)/worker.log 2>&1 & echo "$$!" > $(LOGDIR)/worker.pid
	@echo "Import worker (re)started (log $(LOGDIR)/worker.log)"

stop: ## Stop the backend + frontend + worker dev processes
	@[ -f $(LOGDIR)/backend.pid ]  && kill $$(cat $(LOGDIR)/backend.pid)  2>/dev/null || true
	@[ -f $(LOGDIR)/frontend.pid ] && kill $$(cat $(LOGDIR)/frontend.pid) 2>/dev/null || true
	@[ -f $(LOGDIR)/worker.pid ]   && kill $$(cat $(LOGDIR)/worker.pid)   2>/dev/null || true
	@lsof -ti:8002 | xargs kill 2>/dev/null || true
	@lsof -ti:3002 | xargs kill 2>/dev/null || true
	@pkill -f 'arq daikonstudio.infrastructure.ingestion.worker.WorkerSettings' 2>/dev/null || true
	@rm -f $(LOGDIR)/backend.pid $(LOGDIR)/frontend.pid $(LOGDIR)/worker.pid
	@echo "Dev servers stopped."

logs: ## Tail backend + frontend + worker dev logs
	@mkdir -p $(LOGDIR) && touch $(LOGDIR)/backend.log $(LOGDIR)/frontend.log $(LOGDIR)/worker.log
	tail -f $(LOGDIR)/backend.log $(LOGDIR)/frontend.log $(LOGDIR)/worker.log

generate-api: ## Refresh the OpenAPI snapshot from the backend + regenerate the TS client
	$(BACKEND) && $(BE_ENV) && uv run python -c \
		"import json,sys; from daikonstudio.interface.app import app; sys.stdout.write(json.dumps(app.openapi(), indent=2))" \
		> ../frontend/openapi.json
	$(FRONTEND) && pnpm generate:api

test: ## Backend unit tests + import-linter
	$(BACKEND) && uv run pytest tests/unit -v && uv run lint-imports

test-api: ## Backend API tests
	$(BACKEND) && uv run pytest tests/api -v

test-all: ## All backend tests + import-linter
	$(BACKEND) && uv run pytest -v && uv run lint-imports

test-fe: ## Frontend tests (vitest)
	$(FRONTEND) && pnpm test

lint: ## Backend lint (ruff + mypy)
	$(BACKEND) && uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src

lint-fe: ## Frontend lint (biome)
	$(FRONTEND) && pnpm lint

nuke: ## Stop containers and DELETE all data volumes
	$(COMPOSE) down -v
