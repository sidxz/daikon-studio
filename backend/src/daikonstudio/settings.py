from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STUDIO_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://studio:studio@localhost:5435/studio"
    redis_url: str = "redis://localhost:6381"
    blob_base_url: str = "file:///data/blobs"
    cors_origins: list[str] = ["http://localhost:3002"]
    service_name: str = "daikon-studio"

    # Sentinel (authz mode) — same realm as prot-cellar, chem-cellar, daikon-gen3,
    # docu-store. `sentinel_service_key` defaults to "" as a missing-config signal,
    # not a usable key: an empty key means auth is unconfigured, and the auth
    # dependency (interface/dependencies/_core.py) must reject every request
    # rather than silently let them through.
    sentinel_url: str = "http://localhost:9003"
    sentinel_service_key: str = ""
    idp_jwks_url: str = "https://www.googleapis.com/oauth2/v3/certs"
    idp_audience: str = ""
    idp_issuer: str = "https://accounts.google.com"
