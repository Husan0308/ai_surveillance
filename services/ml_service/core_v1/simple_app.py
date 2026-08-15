from __future__ import annotations

import os
import time
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI

from shared.config import camera_config

from .camera_only_mmap_publisher import CameraOnlyMmapPublisher
from .manager import CameraManager
from .runtime_metrics import process_metrics

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


# Phase 1 invariant:
#   RTSP -> DeepStream/NVDEC -> latest frame -> mmap -> frontend
# There is no detector, tracker, ReID, face recognition or JPEG encoder here.
core_cfg = dict(_load_yaml(ROOT / "config/core_v1.yaml").get("core_v1", {}))
core_cfg["profile"] = "camera-deepstream-mmap-baseline-v1"
core_cfg["capture_backend"] = "deepstream"
core_cfg["drop_on_latency"] = True
core_cfg["postdecode_queue_buffers"] = 1

camera_cfg = camera_config().get("cameras", [])
manager = CameraManager(camera_cfg, core_cfg)

capture_width = int(core_cfg.get("capture_output_width", 736) or 736)
capture_height = int(core_cfg.get("capture_output_height", 416) or 416)
display_fps = float(core_cfg.get("display_fps", 20) or 20)

publishers = {
    cid: CameraOnlyMmapPublisher(
        cid,
        store,
        display_fps,
        capture_width,
        capture_height,
    )
    for cid, store in manager.stores.items()
}

app = FastAPI(title="AI Surveillance Camera Baseline", version="1.0-deepstream-mmap")


@app.on_event("startup")
def startup():
    manager.start()
    stagger = max(0.0, float(core_cfg.get("publisher_start_stagger_ms", 0.0)) / 1000.0)
    for index, publisher in enumerate(publishers.values()):
        publisher.start()
        if stagger and index + 1 < len(publishers):
            time.sleep(stagger)


@app.on_event("shutdown")
def shutdown():
    for publisher in publishers.values():
        publisher.stop()
    for publisher in publishers.values():
        publisher.join()
    manager.stop()


@app.get("/health")
def health():
    camera_metrics = manager.metrics()
    return {
        "status": "ok",
        "mode": "camera-only-deepstream+mmap",
        "profile": core_cfg["profile"],
        "cameras": camera_metrics,
        "online": sum(bool(value.get("online")) for value in camera_metrics.values()),
        "total": len(camera_metrics),
        "detector": {"enabled": False, "ready": False, "reason": "phase_1_baseline"},
        "reid": {"enabled": False, "ready": False, "reason": "phase_1_baseline"},
        "face": {"enabled": False, "ready": False, "reason": "phase_1_baseline"},
        "publishers": {cid: publisher.metrics() for cid, publisher in publishers.items()},
        "service_resources": process_metrics(),
        "display": {
            "width": capture_width,
            "height": capture_height,
            "fps": display_fps,
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
    return {"enabled": False, "cameras": {}, "reason": "phase_1_baseline"}


def run():
    uvicorn.run(
        app,
        host=str(core_cfg.get("ml_host", "0.0.0.0")),
        port=int(core_cfg.get("ml_port", 8001)),
        log_level="info",
    )


if __name__ == "__main__":
    run()
