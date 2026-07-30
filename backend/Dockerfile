# API and default-lane worker. No CUDA, no torch -- the chemprop engine's imports are
# all inside its methods, so this image serves its manifest and renders it in the engine
# picker without being able to run it.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Dependencies before source, so a code change does not re-resolve the world.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra s3 --no-install-project

COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
RUN uv sync --frozen --no-dev --extra s3

ENV PATH="/app/.venv/bin:$PATH"
RUN useradd --create-home --uid 10001 studio && chown -R studio /app
USER studio

# Overridden to `arq daikonstudio.infrastructure.worker.WorkerSettings` for a worker.
CMD ["uvicorn", "daikonstudio.interface.app:app", "--host", "0.0.0.0", "--port", "8000"]
