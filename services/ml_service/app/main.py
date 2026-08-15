from __future__ import annotations

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from services.ml_service.app.config import load_settings
from services.ml_service.app.deepstream.pipeline import DeepStreamRuntime

settings = load_settings()
runtime = DeepStreamRuntime(settings)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    runtime.start()
    try:
        yield
    finally:
        runtime.stop()


app = FastAPI(title="AI Surveillance ML Service", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    snapshot = runtime.snapshot()
    return {
        "service": "ml_service",
        "status": snapshot.state.value,
        "camera_count": snapshot.camera_count,
        "last_error": snapshot.last_error,
    }


@app.get("/cameras")
def cameras() -> dict:
    return {
        "count": len(settings.cameras),
        "cameras": [{"id": camera.camera_id} for camera in settings.cameras],
    }


def main() -> None:
    uvicorn.run(
        "services.ml_service.app.main:app",
        host=os.getenv("ML_HOST", "0.0.0.0"),
        port=int(os.getenv("ML_PORT", "8001")),
        reload=False,
    )


if __name__ == "__main__":
    main()
