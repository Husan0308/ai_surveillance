from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI, HTTPException

from services.camera_v11.monitoring_telemetry_ipc_v1 import MonitoringTelemetryReader
from services.ml_service.app.config import load_settings
from services.ml_service.app.room_pair import router as room_pair_router, state as room_pair_state

settings = load_settings()
camera_ids = tuple(camera.camera_id for camera in settings.cameras)
telemetry = MonitoringTelemetryReader(camera_ids=camera_ids)

app = FastAPI(title="AI Surveillance ML Service", version="0.4.0")
app.include_router(room_pair_router)


@app.get("/health")
def health() -> dict:
    snapshot = telemetry.read()
    readiness = room_pair_state.readiness()
    room_status = str(readiness.get("status", "not_ready"))
    return {
        "service": "ml_service",
        "status": "ok" if room_status == "ready" else "degraded",
        "monitoring_status": snapshot["telemetry_status"],
        "camera_count": len(snapshot["cameras"]),
        "online_camera_count": sum(bool(row["online"]) for row in snapshot["cameras"]),
        "room_pair_readiness": readiness,
    }


@app.get("/cameras")
def cameras() -> dict:
    rows = telemetry.read()["cameras"]
    return {"count": len(rows), "cameras": rows}


@app.get("/api/v1/monitoring/snapshot")
def monitoring_snapshot() -> dict:
    return telemetry.read()


@app.get("/video/{camera_id}")
def video(camera_id: str):
    if camera_id not in camera_ids:
        raise HTTPException(status_code=404, detail=f"Unknown camera: {camera_id}")
    raise HTTPException(
        status_code=503,
        detail=(
            "Video is transported directly from the V11 runtime to Sentinel via "
            "post-OSD shared memory; ml_service does not open or proxy camera streams"
        ),
    )


def main() -> None:
    uvicorn.run(
        app,
        host=os.getenv("ML_HOST", "0.0.0.0"),
        port=int(os.getenv("ML_PORT", "8001")),
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
