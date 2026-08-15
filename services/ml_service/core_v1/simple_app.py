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


def _tune_cameras(items: list[dict]) -> list[dict]:
    """Keep RTSP latency small but large enough to absorb normal LAN jitter."""
    tuned: list[dict] = []
    for raw in items:
        camera = dict(raw)
        if not camera.get("online", True):
            continue
        codec = str(camera.get("display_codec") or camera.get("codec") or "").lower()
        # 20 ms proved too aggressive for a six-stream wall. H.265 generally
        # benefits from a slightly larger jitter window than H.264.
        latency_floor = 80 if codec in {"h265", "hevc"} else 60
        camera["latency_ms"] = max(latency_floor, int(camera.get("latency_ms", latency_floor)))
        camera["rtsp_transport"] = "tcp"
        camera["drop_on_latency"] = True
        tuned.append(camera)
    return tuned


# Phase 1 invariant:
#   RTSP -> DeepStream nvurisrcbin/NVDEC -> newest frame -> mmap -> frontend
# There is no detector, tracker, ReID, face recognition or JPEG encoder here.
core_cfg = dict(_load_yaml(ROOT / "config/core_v1.yaml").get("core_v1", {}))
core_cfg.update(
    {
        "profile": "camera-deepstream-mmap-720p-low-latency-v2",
        "capture_backend": "deepstream",
        "capture_output_width": 1280,
        "capture_output_height": 720,
        "display_fps": 20,
        "rtsp_transport": "tcp",
        "drop_on_latency": True,
        "decoder_extra_surfaces": 6,
        "postdecode_queue_buffers": 1,
        "capture_timeout_ms": 400,
        "max_read_timeouts": 4,
        "startup_grace_sec": 10.0,
        "startup_stagger_sec": 0.25,
        "reconnect_delay_sec": 0.50,
        "capture_metrics_interval_sec": 5.0,
        "max_pipeline_lag_ms": 600,
        "max_pipeline_lag_samples": 20,
    }
)
# The old global 150 ms value hid each camera's own codec-specific latency.
# CameraWorker falls back to camera["latency_ms"] when this key is absent.
core_cfg.pop("rtsp_latency_ms", None)

camera_cfg = _tune_cameras(camera_config().get("cameras", []))
manager = CameraManager(camera_cfg, core_cfg)

capture_width = int(core_cfg["capture_output_width"])
capture_height = int(core_cfg["capture_output_height"])
display_fps = float(core_cfg["display_fps"])

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

app = FastAPI(title="AI Surveillance Camera Baseline", version="2.0-deepstream-720p")


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
            "rtsp_transport": core_cfg["rtsp_transport"],
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
