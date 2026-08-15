from __future__ import annotations

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

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


app = FastAPI(title="AI Surveillance ML Service", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    snapshot = runtime.snapshot()
    return {
        "service": "ml_service",
        "status": snapshot.state.value,
        "camera_count": snapshot.camera_count,
        "online_camera_count": snapshot.online_camera_count,
        "last_error": snapshot.last_error,
    }


@app.get("/cameras")
def cameras() -> dict:
    rows = runtime.camera_metrics()
    return {"count": len(rows), "cameras": rows}


@app.get("/video/{camera_id}")
def video(camera_id: str):
    if not runtime.has_camera(camera_id):
        raise HTTPException(status_code=404, detail=f"Unknown camera: {camera_id}")

    def stream():
        last_version = 0
        try:
            while True:
                jpeg, version = runtime.wait_jpeg(camera_id, last_version, timeout=1.0)
                if jpeg is None or version <= last_version:
                    continue
                last_version = version
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )
        except GeneratorExit:
            return

    return StreamingResponse(
        stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


def main() -> None:
    uvicorn.run(
        "services.ml_service.app.main:app",
        host=os.getenv("ML_HOST", "0.0.0.0"),
        port=int(os.getenv("ML_PORT", "8001")),
        reload=False,
    )


if __name__ == "__main__":
    main()
