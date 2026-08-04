from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STUDIO_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://studio:studio@localhost:5435/studio"
    redis_url: str = "redis://localhost:6381"
    blob_base_url: str = "file:///data/blobs"
    # Forwarded to fsspec's url_to_fs, which is what makes the blob backend swappable
    # with no code change: `endpoint_url` for MinIO or any S3-compatible store,
    # `key`/`secret` for AWS, `account_name`/`connection_string` for Azure Blob. Supplied
    # as JSON in the environment, e.g.
    #   STUDIO_BLOB_STORAGE_OPTIONS='{"endpoint_url": "https://minio-api.example.edu"}'
    blob_storage_options: dict[str, Any] = {}
    # Where pretrained foundation-model weights are cached. Downloaded once on
    # first use; Dockerfile.gpu bakes them in so a production worker never
    # reaches the network, and an air-gapped deployment works.
    pretrained_weights_dir: str = "~/.cache/daikon-studio/weights"
    cors_origins: list[str] = ["http://localhost:3002"]
    service_name: str = "daikon-studio"

    # Selects InlineEnqueuer over ArqEnqueuer (infrastructure/worker.py) --
    # runs jobs in-process so tests and local dev need no Valkey at all.
    inline_jobs: bool = False

    # Which lane's queue this worker process pulls. Engines declare a lane on their
    # manifest and a deployment satisfies it by running a worker here -- nothing in the
    # codebase names a host. See
    # docs/superpowers/specs/2026-07-30-remote-engines-chemprop-design.md.
    worker_lane: str = "default"
    # arq's own default is 10. A GPU worker MUST set this to 1 (or run one process per
    # device with CUDA_VISIBLE_DEVICES pinned): concurrent fits on one device exhaust
    # its memory, and arq will happily start ten.
    worker_max_jobs: int = 10
    # The SOFT deadline, in seconds, enforced cooperatively inside the engine through
    # TrainContext.report. arq's hard job_timeout is derived from this with a margin;
    # see infrastructure/worker.py. Raise this on a GPU lane, not the hard timeout.
    worker_job_timeout: int = 1800

    # Sentinel (authz mode) — same realm as prot-cellar, chem-cellar, daikon-gen3,
    # docu-store. `sentinel_service_key` defaults to "" as a missing-config signal,
    # not a usable key: an empty key means auth is unconfigured, and the auth
    # dependency (interface/dependencies/_core.py) must reject every request
    # rather than silently let them through.
    sentinel_url: str = "http://localhost:9003"
    sentinel_service_key: str = ""
    # The identity Sentinel knows this deployment by, which is NOT always
    # `service_name` -- that one is display text for /version. A dev instance is
    # commonly registered under its own name (e.g. "daikon-studio-dev") so its
    # actions and role grants are scoped separately from production's within the
    # shared realm. Empty falls back to `service_name`, so a deployment whose
    # registered name matches the display name needs no extra config. prot-cellar
    # keeps these separate too (SENTINEL_SERVICE_NAME); conflating them silently
    # registers actions under one identity and checks them under another.
    sentinel_service_name: str = ""
    idp_jwks_url: str = "https://www.googleapis.com/oauth2/v3/certs"
    idp_audience: str = ""
    idp_issuer: str = "https://accounts.google.com"
