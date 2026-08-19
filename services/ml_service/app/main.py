from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from services.ml_service.app.config import load_settings

settings = load_settings()
runtime: Any | None = None


def _runtime():
    if runtime is None:
        raise RuntimeError("ml_service runtime is not started")
    return runtime


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global runtime
    # Import the camera/DeepStream runtime only in the real server process.
    # A spawned detector child re-imports this module as __mp_main__; keeping
    # this import here prevents the CUDA child from constructing camera state.
    from services.ml_service.app.deepstream.pipeline import DeepStreamRuntime

    runtime = DeepStreamRuntime(settings)
    runtime.start()
    try:
        yield
    finally:
        current = runtime
        runtime = None
        if current is not None:
            current.stop()


app = FastAPI(title="AI Surveillance ML Service", version="0.7.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    current = _runtime()
    snapshot = current.snapshot()
    return {
        "service": "ml_service",
        "status": snapshot.state.value,
        "camera_count": snapshot.camera_count,
        "online_camera_count": snapshot.online_camera_count,
        "detector": current.detector_metrics(),
        "tracker": current.tracker_metrics(),
        "last_error": snapshot.last_error,
    }


@app.get("/cameras")
def cameras() -> dict:
    rows = _runtime().camera_metrics()
    return {"count": len(rows), "cameras": rows}


@app.get("/detections/{camera_id}")
def detections(camera_id: str) -> dict:
    current = _runtime()
    if not current.has_camera(camera_id):
        raise HTTPException(status_code=404, detail=f"Unknown camera: {camera_id}")
    return current.detection_payload(camera_id)


@app.get("/tracks")
def all_tracks() -> dict:
    current = _runtime()
    rows = []
    for camera in settings.cameras:
        payload = current.tracking_payload(camera.camera_id)
        result = payload.get("result")
        if result is None:
            result = {
                "camera_id": camera.camera_id,
                "frame_id": 0,
                "people": 0,
                "age_ms": None,
                "tracks": [],
            }
        rows.append(result)
    return {
        "count": len(rows),
        "tracker": current.tracker_metrics(),
        "tracks": rows,
    }


@app.get("/tracks/{camera_id}")
def tracks(camera_id: str) -> dict:
    current = _runtime()
    if not current.has_camera(camera_id):
        raise HTTPException(status_code=404, detail=f"Unknown camera: {camera_id}")
    return current.tracking_payload(camera_id)


@app.get("/video/{camera_id}")
def video(camera_id: str):
    current = _runtime()
    if not current.has_camera(camera_id):
        raise HTTPException(status_code=404, detail=f"Unknown camera: {camera_id}")

    def stream():
        last_version = 0
        try:
            while True:
                jpeg, version = current.wait_jpeg(camera_id, last_version, timeout=1.0)
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
        app,
        host=os.getenv("ML_HOST", "0.0.0.0"),
        port=int(os.getenv("ML_PORT", "8001")),
        reload=False,
    )


if __name__ == "__main__":
    main()
