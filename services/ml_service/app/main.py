from __future__ import annotations

import hmac
import math
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from services.shared.deployment import analytics_enabled, load_deployment
deployment = load_deployment()  # Service settings precede legacy credential dotenv.

from services.camera_v11.monitoring_telemetry_ipc_v1 import MonitoringTelemetryReader
from services.camera_v11.tracked_object_ipc_v1 import read_snapshot as read_tracked_objects
from services.ml_service.app.config import load_settings
from services.ml_service.app.room_pair import router as room_pair_router, state as room_pair_state

settings = load_settings()
camera_ids = tuple(camera.camera_id for camera in settings.cameras)
telemetry = MonitoringTelemetryReader(camera_ids=camera_ids)

app = FastAPI(title="AI Surveillance ML Service", version="0.4.0")
app.include_router(room_pair_router)

_REQUIRED_CAMERA_STAGES = (
    "age_since_last_decoded_frame_ms",
    "age_since_last_pgie_frame_ms",
    "age_since_last_tracker_output_ms",
    "age_since_last_preview_ms",
)
_CAMERA_FRESHNESS_LIMIT_MS = 2000.0


def _camera_stages_fresh(row: dict) -> bool:
    """Fail closed if any required per-source stage is missing or stale."""
    if row.get("online") is not True or row.get("current_state") != "ONLINE":
        return False
    for stage in _REQUIRED_CAMERA_STAGES:
        value = row.get(stage)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or value > _CAMERA_FRESHNESS_LIMIT_MS):
            return False
    return True


def _authorized(header: str | None) -> bool:
    token = deployment.access_token
    if not token:
        return True
    scheme, _, supplied = (header or "").partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(supplied, token)


@app.middleware("http")
async def authenticate_ml_requests(request: Request, call_next):
    if request.url.path != "/health" and not _authorized(request.headers.get("authorization")):
        return JSONResponse({"detail": "unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED,
                            headers={"WWW-Authenticate": "Bearer"})
    return await call_next(request)


@app.get("/health")
def health() -> dict:
    snapshot = telemetry.read()
    readiness = room_pair_state.readiness() if analytics_enabled() else {"status": "disabled", "ready": False}
    room_status = str(readiness.get("status", "not_ready"))
    runtime = snapshot.get("runtime", {})
    camera_rows = snapshot.get("cameras", [])
    six_camera_ready = (
        snapshot["telemetry_status"] == "fresh"
        and len(camera_rows) == len(camera_ids) == 6
        and tuple(row.get("camera_id") for row in camera_rows) == camera_ids
        and all(_camera_stages_fresh(row) for row in camera_rows)
        and runtime.get("shared_pipeline_status") == "healthy"
        and runtime.get("detector_enabled") is True
    )
    if snapshot["telemetry_status"] == "offline":
        analytics_status = "unavailable"
    elif snapshot["telemetry_status"] != "fresh":
        analytics_status = "degraded"
    elif runtime.get("detector_enabled") is False:
        analytics_status = "disabled"
    else:
        analytics_status = "ready" if six_camera_ready else "degraded"
    return {
        "service": "ml_service",
        "status": "ok",
        "process_status": "running",
        "analytics_enabled": analytics_enabled(),
        "monitoring_status": snapshot["telemetry_status"],
        "analytics_status": analytics_status,
        "camera_count": len(snapshot["cameras"]),
        "online_camera_count": sum(row.get("online") is True for row in snapshot["cameras"]),
        "six_camera_readiness": "ready" if six_camera_ready else "not_ready",
        "shared_pipeline_status": runtime.get("shared_pipeline_status", "unknown"),
        "cameras": camera_rows,
        "room_pair_readiness": readiness,
    }


@app.get("/cameras")
def cameras() -> dict:
    rows = telemetry.read()["cameras"]
    return {"count": len(rows), "cameras": rows}


@app.get("/api/v1/monitoring/snapshot")
def monitoring_snapshot() -> dict:
    return telemetry.read()


@app.get("/api/v1/tracked-objects")
def tracked_objects() -> dict:
    """Expose the same latest-only native post-NvDCF feed consumed by the UI."""
    return read_tracked_objects(camera_ids=camera_ids)


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
        host=deployment.ml_host,
        port=deployment.ml_port,
        reload=False,
        access_log=False,
        timeout_graceful_shutdown=2,
    )


if __name__ == "__main__":
    main()
