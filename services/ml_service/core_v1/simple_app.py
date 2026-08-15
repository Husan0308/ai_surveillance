from __future__ import annotations

import os
import time
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, StreamingResponse

from shared.config import camera_config

from .event_publisher import EventDrivenJpegPublisher
from .manager import CameraManager
from .runtime_metrics import process_metrics
from .stable_detector import StableYoloDetectorWorker

ROOT = Path(__file__).resolve().parents[3]


def _expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return _expand(yaml.safe_load(handle) or {})


# Keep the full project config untouched for later. Simple mode overrides only
# presentation/capture settings and intentionally does not start ReID or Face.
core_cfg = dict(_load_yaml(ROOT / "config/core_v1.yaml").get("core_v1", {}))
core_cfg.update(
    {
        "profile": "simple-clear-detection-v1",
        "display_fps": 20,
        "jpeg_quality": 88,
        "max_display_width": 960,
        "max_display_height": 540,
        "capture_output_width": 960,
        "capture_output_height": 540,
    }
)

camera_cfg = camera_config().get("cameras", [])
manager = CameraManager(camera_cfg, core_cfg)

detector_cfg = dict(core_cfg.get("detector") or {})
detector = StableYoloDetectorWorker(manager.stores, detector_cfg, ROOT)

# Base event-driven publisher deliberately has no identity provider. It still
# smooths detector observations with the visual tracker, but the overlay label
# stays simply "Person" rather than Cxx/Gxxx/name identity UI.
publishers = {
    cid: EventDrivenJpegPublisher(
        cid,
        store,
        core_cfg.get("display_fps", 20),
        core_cfg.get("jpeg_quality", 88),
        core_cfg.get("max_display_width", 960),
        core_cfg.get("max_display_height", 540),
        detections=detector.results,
        overlay_max_age_ms=detector_cfg.get("overlay_max_age_ms", 700),
        tracker_config=core_cfg.get("visual_tracker") or {},
        identity_provider=None,
    )
    for cid, store in manager.stores.items()
}

app = FastAPI(title="AI Surveillance Simple Detection Core", version="1.0-simple")


@app.on_event("startup")
def startup():
    manager.start()
    stagger = max(0.0, float(core_cfg.get("publisher_start_stagger_ms", 0.0)) / 1000.0)
    for index, publisher in enumerate(publishers.values()):
        publisher.start()
        if stagger and index + 1 < len(publishers):
            time.sleep(stagger)
    detector.start()


@app.on_event("shutdown")
def shutdown():
    detector.stop()
    detector.join(10)
    for publisher in publishers.values():
        publisher.stop()
    for publisher in publishers.values():
        publisher.join()
    manager.stop()


@app.get("/health")
def health():
    camera_metrics = manager.metrics()
    detector_metrics = detector.metrics()
    return {
        "status": "ok" if detector_metrics.get("ready") else "degraded",
        "mode": "simple-camera+detection",
        "profile": core_cfg["profile"],
        "cameras": camera_metrics,
        "online": sum(bool(value.get("online")) for value in camera_metrics.values()),
        "total": len(camera_metrics),
        "detector": detector_metrics,
        "reid": {"enabled": False, "ready": False, "reason": "simple_mode"},
        "face": {"enabled": False, "ready": False, "reason": "simple_mode"},
        "publishers": {cid: publisher.metrics() for cid, publisher in publishers.items()},
        "service_resources": process_metrics(),
        "display": {
            "width": int(core_cfg["max_display_width"]),
            "height": int(core_cfg["max_display_height"]),
            "jpeg_quality": int(core_cfg["jpeg_quality"]),
            "fps": int(core_cfg["display_fps"]),
        },
    }


@app.get("/cameras")
def cameras():
    metrics = manager.metrics()
    return [{"id": cid, **metrics[cid]} for cid in sorted(metrics)]


@app.get("/detections")
def detections():
    now = time.monotonic()
    results = {}
    for cid, result in detector.results.snapshot().items():
        results[cid] = {
            "frame_id": result.frame_id,
            "result_age_ms": max(0.0, (now - result.produced_monotonic) * 1000.0),
            "capture_age_ms": max(0.0, (now - result.frame_captured_monotonic) * 1000.0),
            "boxes": [
                {
                    "bbox": [box.x1, box.y1, box.x2, box.y2],
                    "confidence": box.confidence,
                }
                for box in result.boxes
            ],
        }
    return {"enabled": True, "cameras": results, "metrics": detector.metrics()}


@app.get("/frame/{camera_id}")
def latest_frame(camera_id: str, after: int = Query(-1), wait_ms: int = Query(200, ge=0, le=500)):
    if camera_id not in publishers:
        raise HTTPException(404, "camera not found")
    jpeg, version, published, source_frame_id = publishers[camera_id].wait_newer(after, wait_ms / 1000.0)
    if jpeg is None:
        raise HTTPException(503, "frame not ready")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Frame-Version": str(version),
            "X-Source-Frame-Id": str(source_frame_id),
            "X-Published-Monotonic": f"{published:.6f}",
        },
    )


def _mjpeg(camera_id: str):
    publisher = publishers[camera_id]
    last = -1
    while True:
        jpeg, version, _, _ = publisher.wait_newer(last, 0.5)
        if jpeg is None or version <= last:
            continue
        last = version
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(jpeg)).encode()
            + b"\r\n\r\n"
            + jpeg
            + b"\r\n"
        )


@app.get("/video/{camera_id}")
def video(camera_id: str):
    if camera_id not in publishers:
        raise HTTPException(404, "camera not found")
    return StreamingResponse(_mjpeg(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")


def run():
    uvicorn.run(
        app,
        host=str(core_cfg.get("ml_host", "0.0.0.0")),
        port=int(core_cfg.get("ml_port", 8001)),
        log_level="info",
    )


if __name__ == "__main__":
    run()
