from fastapi import FastAPI

from daikonstudio.settings import Settings


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(title="daikon-studio", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    async def version() -> dict[str, str]:
        return {"service": settings.service_name, "version": app.version}

    return app
