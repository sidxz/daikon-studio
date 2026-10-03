# API and default-lane worker. No CUDA, no torch -- the chemprop engine's imports are
# all inside its methods, so this image serves its manifest and renders it in the engine
# picker without being able to run it.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# libgomp1 is load-bearing. LightGBM's wheel links the system OpenMP runtime
# (libgomp.so.1) and python:3.13-slim does not ship it, so without this line the
# image builds green and the API dies at import with
# `OSError: libgomp.so.1: cannot open shared object file`. Found 2026-10-02 by
# booting the image, not by building it -- the same trap docs/roadmap.md records
# for the GPU image. `make image-smoke` and CI now run that boot check.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Dependencies before source, so a code change does not re-resolve the world.
COPY pyproject.toml uv.lock ./
# The uv cache lives in a BuildKit cache mount, not in the layer: with UV_LINK_MODE=copy
# a cached package would otherwise be stored twice (cache + .venv). On the CI runner that
# duplication ran the API image out of disk once numba/llvmlite (umap-learn) joined.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra s3 --no-install-project

COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra s3

ENV PATH="/app/.venv/bin:$PATH"
# /data/blobs is the default STUDIO_BLOB_BASE_URL. Created and owned here so a
# named volume mounted on it inherits writable ownership for uid 10001 instead
# of failing the first upload with PermissionError.
RUN useradd --create-home --uid 10001 studio \
 && mkdir -p /data/blobs \
 && chown -R studio /app /data
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

# For a worker, override CMD to `["python", "-m", "daikonstudio.infrastructure.runner"]`
# and set env vars STUDIO_URL (server address) and STUDIO_RUNNER_TOKEN (auth token).
# Build this image (tagged daikon-runner:cpu, the tag the runners UI's docker command
# names) with `make image-runner-cpu`; Dockerfile.gpu's own CMD is the agent already,
# built with `make image-runner-gpu`.
CMD ["uvicorn", "daikonstudio.interface.app:app", "--host", "0.0.0.0", "--port", "8000"]
