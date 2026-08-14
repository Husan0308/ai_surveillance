from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, StreamingResponse

from shared.config import camera_config
from services.ml_service.heatmap import CameraAnkleHeatmapCoordinator
from services.ml_service.pose import PoseCoordinator

from .heatmap_publisher import HeatmapJpegPublisher
from .manager import CameraManager
from .reid_service import ReIDCoordinator
from .runtime_metrics import process_metrics
from .spatial_calibration import RoomSpatialMapper
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


camera_cfg = camera_config().get("cameras", [])
core_cfg = _load_yaml(ROOT / "config/core_v1.yaml").get("core_v1", {})
manager = CameraManager(camera_cfg, core_cfg)

detector_cfg = dict(core_cfg.get("detector") or {})
visual_cfg = dict(core_cfg.get("visual_tracker") or {})
pose_cfg = dict(core_cfg.get("pose") or {})
heatmap_cfg = dict(core_cfg.get("heatmap") or {})
reid_cfg = dict(core_cfg.get("reid") or {})

spatial_mapper = RoomSpatialMapper(ROOT / "config/room_mapping.yaml")

detector = (
    StableYoloDetectorWorker(manager.stores, detector_cfg, ROOT)
    if bool(detector_cfg.get("enabled", False))
    else None
)

reid = (
    ReIDCoordinator(
        manager.stores,
        detector.results,
        reid_cfg,
        spatial_mapper=spatial_mapper,
    )
    if detector is not None and bool(reid_cfg.get("enabled", False))
    else None
)

# Pose is isolated in a spawned CPU process. A fatal native signal in
# Ultralytics/PyTorch therefore cannot abort this Uvicorn/camera process.
pose = (
    PoseCoordinator(manager.stores, detector.results, pose_cfg)
    if detector is not None and bool(pose_cfg.get("enabled", False))
    else None
)

# Camera-space heatmap consumes the pose result stream directly. It does not
# require room-floor homography and is rendered on the camera JPEG itself.
heatmap = (
    CameraAnkleHeatmapCoordinator(pose, manager.stores, heatmap_cfg)
    if pose is not None and bool(heatmap_cfg.get("enabled", False))
    else None
)

publishers = {
    cid: HeatmapJpegPublisher(
        cid,
        store,
        core_cfg.get("display_fps", 12),
        core_cfg.get("jpeg_quality", 82),
        core_cfg.get("max_display_width", 960),
        core_cfg.get("max_display_height", 540),
        detections=(detector.results if detector else None),
        overlay_max_age_ms=detector_cfg.get("overlay_max_age_ms", 350),
        tracker_config=visual_cfg,
        identity_provider=reid,
        heatmap_provider=heatmap,
    )
    for cid, store in manager.stores.items()
}

app = FastAPI(title="AI Surveillance ML Core v1", version="2.1-camera-heatmap")
_optional_stop = threading.Event()
_optional_thread = None


def _mode() -> str:
    parts = ["camera"]
    if detector:
        parts.append("yolo")
    if reid:
        parts.append("reid")
    if pose:
        parts.append("pose-process")
    if heatmap:
        parts.append("camera-heatmap")
    return "+".join(parts)


def _start_optional_after_detector():
    # Avoid simultaneous model initialization. Detection gets the machine first;
    # optional analytics start only after detector ready, and remain non-gating.
    deadline = time.monotonic() + max(
        5.0, float(core_cfg.get("optional_start_timeout_sec", 45.0))
    )
    while not _optional_stop.is_set():
        if detector is None:
            break
        try:
            if bool(detector.metrics().get("ready")):
                break
        except Exception:
            pass
        if time.monotonic() >= deadline:
            return
        _optional_stop.wait(0.10)
    if _optional_stop.is_set():
        return
    if reid:
        reid.start()
    if pose:
        pose.start()
    if heatmap:
        heatmap.start()


@app.on_event("startup")
def startup():
    global _optional_thread
    _optional_stop.clear()
    manager.start()

    stagger = max(
        0.0,
        float(core_cfg.get("publisher_start_stagger_ms", 0.0)) / 1000.0,
    )
    for index, publisher in enumerate(publishers.values()):
        publisher.start()
        if stagger and index + 1 < len(publishers):
            time.sleep(stagger)

    if detector:
        detector.start()

    _optional_thread = threading.Thread(
        target=_start_optional_after_detector,
        name="optional-analytics-starter",
        daemon=False,
    )
    _optional_thread.start()


@app.on_event("shutdown")
def shutdown():
    _optional_stop.set()
    if _optional_thread:
        _optional_thread.join(2)
    if heatmap:
        heatmap.stop()
        heatmap.join(4)
    if pose:
        pose.stop()
        pose.join(6)
    if reid:
        reid.stop()
        reid.join(6)
    if detector:
        detector.stop()
        detector.join(10)

    for publisher in publishers.values():
        publisher.stop()
    for publisher in publishers.values():
        publisher.join()
    manager.stop()


@app.get("/health")
def health():
    metrics = manager.metrics()
    detector_metrics = detector.metrics() if detector else None
    detector_ready = bool(detector_metrics and detector_metrics.get("ready"))
    return {
        "status": "ok" if detector_ready or detector is None else "degraded",
        "mode": _mode(),
        "profile": "camera-ankle-heatmap-crash-isolated",
        "cameras": metrics,
        "online": sum(bool(value.get("online")) for value in metrics.values()),
        "total": len(metrics),
        "detector": detector_metrics,
        "pose": pose.metrics() if pose else {"enabled": False},
        "heatmap": heatmap.snapshot() if heatmap else {"enabled": False},
        "reid": reid.metrics() if reid else {"enabled": False},
        "publishers": {
            cid: publisher.metrics() for cid, publisher in publishers.items()
        },
        "frame_history": {
            cid: store.history_metrics()
            for cid, store in manager.stores.items()
            if hasattr(store, "history_metrics")
        },
        "service_resources": process_metrics(),
    }


@app.get("/cameras")
def cameras():
    metrics = manager.metrics()
    return [{"id": cid, **metrics[cid]} for cid in sorted(metrics)]


@app.get("/detections")
def detections():
    if detector is None:
        return {"enabled": False, "cameras": {}}
    now = time.monotonic()
    results = {}
    for cid, result in detector.results.snapshot().items():
        results[cid] = {
            "frame_id": result.frame_id,
            "result_age_ms": max(
                0.0, (now - result.produced_monotonic) * 1000.0
            ),
            "capture_age_ms": max(
                0.0, (now - result.frame_captured_monotonic) * 1000.0
            ),
            "boxes": [
                {
                    "bbox": [box.x1, box.y1, box.x2, box.y2],
                    "confidence": box.confidence,
                }
                for box in result.boxes
            ],
        }
    return {"enabled": True, "cameras": results, "metrics": detector.metrics()}


@app.get("/poses")
def poses_state():
    if pose is None:
        return {"enabled": False, "cameras": {}}
    now = time.monotonic()
    results = {}
    for cid, result in pose.snapshot().items():
        results[cid] = {
            "frame_id": result.frame_id,
            "result_age_ms": max(
                0.0, (now - result.produced_monotonic) * 1000.0
            ),
            "capture_age_ms": max(
                0.0, (now - result.frame_captured_monotonic) * 1000.0
            ),
            "people": [
                {
                    "bbox": list(person.bbox),
                    "confidence": person.confidence,
                    "keypoints": [
                        {
                            "x": point.x,
                            "y": point.y,
                            "confidence": point.confidence,
                        }
                        for point in person.keypoints
                    ],
                }
                for person in result.people
            ],
        }
    return {"enabled": True, "cameras": results, "metrics": pose.metrics()}


@app.get("/heatmap")
def heatmap_state():
    if heatmap is None:
        return {"enabled": False, "cameras": {}}
    return heatmap.snapshot()


@app.post("/heatmap/reset/{camera_id}")
def reset_heatmap(camera_id: str):
    if heatmap is None:
        raise HTTPException(503, "camera heatmap is disabled")
    try:
        heatmap.reset(camera_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True, "camera_id": camera_id}


@app.post("/heatmap/reset")
def reset_all_heatmaps():
    if heatmap is None:
        raise HTTPException(503, "camera heatmap is disabled")
    heatmap.reset()
    return {"ok": True}


@app.get("/reid")
def reid_state():
    if reid is None:
        return {"enabled": False}
    return {"enabled": True, "state": reid.snapshot(), "metrics": reid.metrics()}


@app.get("/room-mapping")
def room_mapping():
    payload = spatial_mapper.snapshot()
    payload["people"] = reid.room_people() if reid is not None else []
    payload["heatmap"] = {
        "mode": "camera_pixels",
        "note": "heatmap is rendered directly on each camera frame from ankle keypoints",
    }
    return payload


@app.post("/room-mapping/calibrate")
def calibrate_room_camera(payload: dict):
    try:
        return {
            "ok": True,
            "calibration": spatial_mapper.calibrate(
                payload.get("camera_id"),
                payload.get("image_points") or [],
                payload.get("room_points") or [],
                payload.get("image_size"),
                method="assisted",
            ),
        }
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/room-mapping/reset/{camera_id}")
def reset_room_camera(camera_id: str):
    try:
        return {"ok": True, "calibration": spatial_mapper.clear_calibration(camera_id)}
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/room-mapping/auto-discovery")
def automatic_room_pair(payload: dict):
    left = str(payload.get("left_camera") or "")
    right = str(payload.get("right_camera") or "")
    if (left, right) not in spatial_mapper.camera_pairs() and (
        right,
        left,
    ) not in spatial_mapper.camera_pairs():
        raise HTTPException(400, "camera pair is not a verified same-room pair")

    left_frame = manager.stores[left].get()[0] if left in manager.stores else None
    right_frame = manager.stores[right].get()[0] if right in manager.stores else None
    evidence = spatial_mapper.automatic_pair_evidence(
        getattr(left_frame, "image", None),
        getattr(right_frame, "image", None),
    )
    evidence.update(
        {
            "left_camera": left,
            "right_camera": right,
            "persisted_as_floor_calibration": False,
        }
    )
    return evidence


@app.get("/frame/{camera_id}")
def latest_frame(
    camera_id: str,
    after: int = Query(-1),
    wait_ms: int = Query(200, ge=0, le=500),
):
    if camera_id not in publishers:
        raise HTTPException(404, "camera not found")
    publisher = publishers[camera_id]
    jpeg, version, published, source_frame_id = publisher.wait_newer(
        after,
        wait_ms / 1000.0,
    )
    if jpeg is None:
        raise HTTPException(503, "frame not ready")
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "X-Frame-Version": str(version),
        "X-Source-Frame-Id": str(source_frame_id),
        "X-Published-Monotonic": f"{published:.6f}",
    }
    return Response(content=jpeg, media_type="image/jpeg", headers=headers)


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
    return StreamingResponse(
        _mjpeg(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
