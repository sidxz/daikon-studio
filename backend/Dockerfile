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
# OMP_NUM_THREADS=1 is load-bearing, not tuning. torch and scikit-learn each ship
# their own OpenMP runtime, and a training job loads BOTH -- RunTraining fits the
# chosen engine and the mandatory ECFP4 baseline in one process, by design. Multiple
# OpenMP runtimes in one process is undefined behaviour; on macOS it segfaults in
# __kmp_fork_barrier partway through a real chemprop fit. Setting it here so the
# deployed worker cannot hit the same wall hours into a GPU run.
# ponytail: costs intra-op parallelism in the baseline fit. Raise it only with a
# measurement showing this platform does not have the duplicate-runtime problem.
ENV OMP_NUM_THREADS=1

USER studio

# Overridden to `arq daikonstudio.infrastructure.worker.WorkerSettings` for a worker.
CMD ["uvicorn", "daikonstudio.interface.app:app", "--host", "0.0.0.0", "--port", "8000"]
