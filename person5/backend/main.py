from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .analysis import router as analysis_router
from .auth import router as auth_router
from .upload import router as upload_router


import os

def create_app() -> FastAPI:
    app = FastAPI(title="SatQuery API", version="0.2.0")

    cors_origins = os.getenv("SATQUERY_CORS_ORIGINS", "*").split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "stage": "integrated", "providers": "configured-by-environment"}

    app.include_router(auth_router)
    app.include_router(upload_router)
    app.include_router(analysis_router)
    return app


app = create_app()
