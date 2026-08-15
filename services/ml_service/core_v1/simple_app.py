from __future__ import annotations

import os
import time
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI

from shared.config import camera_config

from .manager import CameraManager
from .mmap_publisher import MmapFramePublisher
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


# Simple mode keeps the expensive identity stack frozen. Camera -> detection ->
# local mmap display is the entire hot path. The detector still receives its
# configured 416x736 tensor, independent from the 960x540 presentation frame.
core_cfg = dict(_load_yaml(ROOT / "config/core_v1.yaml").get("core_v1", {}))
core_cfg.update(
    {
        "profile": "simple-smooth-detection-mmap-v2",
        "display_fps": 20,
        "max_display_width": 960,
        "max_display_height": 540,
        "capture_output_width": 960,
        "capture_output_height": 540,
        "drop_on_latency": True,
        "postdecode_queue_buffers": 1,
    }
)
# Do not force the old global 150 ms jitterbuffer in simple mode. Every camera
# already has a tuned latency in cameras.yaml (20 ms H264, 80 ms H265), and
# CameraWorker falls back to that value when the global override is absent.
core_cfg.pop("rtsp_latency_ms", None)

camera_cfg = camera_config().get("cameras", [])
manager = CameraManager(camera_cfg, core_cfg)

detector_cfg = dict(core_cfg.get("detector") or {})
detector = StableYoloDetectorWorker(manager.stores, detector_cfg, ROOT)

publishers = {
    cid: MmapFramePublisher(
        cid,
        store,
        core_cfg.get("display_fps", 20),
        0,  # JPEG is intentionally not used in simple local mode.
        core_cfg.get("max_display_width", 960),
        core_cfg.get("max_display_height", 540),
        detections=detector.results,
        overlay_max_age_ms=detector_cfg.get("overlay_max_age_ms", 700),
        tracker_config=core_cfg.get("visual_tracker") or {},
        identity_provider=None,
    )
    for cid, store in manager.stores.items()
}

app = FastAPI(title="AI Surveillance Smooth Detection Core", version="2.0-mmap")


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
        "mode": "simple-camera+detection+mmap",
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
            "fps": int(core_cfg["display_fps"]),
            "transport": "mmap-bgr-double-buffer",
            "codec": "none",
            "http_mjpeg": False,
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


def run():
    uvicorn.run(
        app,
        host=str(core_cfg.get("ml_host", "0.0.0.0")),
        port=int(core_cfg.get("ml_port", 8001)),
        log_level="info",
    )


if __name__ == "__main__":
    run()
