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

    # Selects InlineEnqueuer over DbEnqueuer (infrastructure/jobs.py) --
    # runs jobs in-process so tests and local dev need no runner at all.
    inline_jobs: bool = False

    # Which lane's queue this worker process pulls. Engines declare a lane on their
    # manifest and a deployment satisfies it by running a worker here -- nothing in the
    # codebase names a host. See
    # docs/superpowers/specs/2026-07-30-remote-engines-chemprop-design.md.
    worker_lane: str = "default"
    # A GPU worker MUST set this to 1 (or run one process per device with
    # CUDA_VISIBLE_DEVICES pinned): concurrent fits on one device exhaust its memory,
    # and an unbounded poller will happily start ten.
    worker_max_jobs: int = 10
    # The SOFT deadline, in seconds, enforced cooperatively inside the engine through
    # TrainContext.report. Raise this on a GPU lane.
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

    # Self-hosted runners (2026-08-04 spec). Gates `RunQueue.claim_next`'s per-workspace
    # concurrency cap: no single workspace can starve every other tenant's runs off a
    # shared runner fleet.
    workspace_max_active_runs: int = 10
    # How long a runner's claim on a run holds before `RunQueue.sweep` requeues it --
    # covers a runner that crashes or loses connectivity mid-job. Consumed by
    # `claim_next`/`verify_claim`'s `lease_seconds` and extended on every heartbeat.
    runner_lease_seconds: int = 600
    # How many times `claim_next` will hand the same run to a (possibly different)
    # runner before `sweep` gives up and fails it outright -- caps a poison-pill job
    # from cycling through the fleet forever.
    runner_max_attempts: int = 3
    # Ceiling on a runner's artifact/log upload for one run, enforced by the upload
    # endpoint a later task adds. Default 1 GiB.
    runner_upload_max_bytes: int = 1_073_741_824
    # How recently a runner must have heartbeated (`touch_last_seen`) to show as
    # `online` in `ListRunners` -- see `application/runners/manage.py`.
    runner_online_threshold_seconds: int = 15
