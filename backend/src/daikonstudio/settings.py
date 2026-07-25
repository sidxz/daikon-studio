from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STUDIO_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://studio:studio@localhost:5435/studio"
    redis_url: str = "redis://localhost:6381"
    blob_base_url: str = "file:///data/blobs"
    cors_origins: list[str] = ["http://localhost:3002"]
    service_name: str = "daikon-studio"
