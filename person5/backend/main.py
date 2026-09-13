from fastapi import FastAPI

from .analysis import router as analysis_router
from .auth import router as auth_router
from .upload import router as upload_router


def create_app() -> FastAPI:
    app = FastAPI(title="SatQuery API", version="0.2.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "stage": "integrated", "providers": "configured-by-environment"}

    app.include_router(auth_router)
    app.include_router(upload_router)
    app.include_router(analysis_router)
    return app


app = create_app()
