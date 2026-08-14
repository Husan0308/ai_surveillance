from __future__ import annotations

import os
import time
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from shared.config import camera_config

from .face_service_safe import SafeFaceRecognitionService
from .manager import CameraManager
from .room_consensus_reid import RoomConsensusGlobalReIdCoordinator
from .runtime_metrics import process_metrics
from .stable_detector import StableYoloDetectorWorker
from .tracking_publisher import TrackingJpegPublisher

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


class FaceEnrollRequest(BaseModel):
    name: str
    department: str = ""
    employee_id: str = ""
    sample_tokens: list[str]


camera_cfg = camera_config().get("cameras", [])
core_cfg = _load_yaml(ROOT / "config/core_v1.yaml").get("core_v1", {})
manager = CameraManager(camera_cfg, core_cfg)

detector_cfg = dict(core_cfg.get("detector") or {})
visual_cfg = dict(core_cfg.get("visual_tracker") or {})
reid_cfg = dict(core_cfg.get("reid") or {})
face_cfg = dict(core_cfg.get("face") or {})

detector = (
    StableYoloDetectorWorker(manager.stores, detector_cfg, ROOT)
    if bool(detector_cfg.get("enabled", True))
    else None
)

publishers = {
    cid: TrackingJpegPublisher(
        cid,
        store,
        core_cfg.get("display_fps", 15),
        core_cfg.get("jpeg_quality", 72),
        core_cfg.get("max_display_width", 640),
        core_cfg.get("max_display_height", 360),
        detections=(detector.results if detector else None),
        overlay_max_age_ms=detector_cfg.get("overlay_max_age_ms", 700),
        tracker_config=visual_cfg,
        identity_provider=None,
    )
    for cid, store in manager.stores.items()
}

reid = (
    RoomConsensusGlobalReIdCoordinator(manager.stores, publishers, reid_cfg, ROOT)
    if bool(reid_cfg.get("enabled", False))
    else None
)

face = (
    SafeFaceRecognitionService(
        manager.stores,
        publishers,
        face_cfg,
        ROOT,
        base_identity=reid,
    )
    if bool(face_cfg.get("enabled", False))
    else None
)

identity_provider = face or reid
if identity_provider is not None:
    for publisher in publishers.values():
        publisher.identity_provider = identity_provider

app = FastAPI(
    title="AI Surveillance Detection + Tracking + ReID + Face Core",
    version="1.5-face",
)


def _mode() -> str:
    if detector is None:
        return "camera-only"
    parts = ["camera", "yolo-detect", "local-track"]
    if reid is not None:
        parts.append("room-consensus-reid")
    if face is not None:
        parts.append("face-cpu")
    return "+".join(parts)


@app.on_event("startup")
def startup():
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
    if reid:
        reid.start()
    if face:
        face.start()


@app.on_event("shutdown")
def shutdown():
    if face:
        face.stop()
        face.join(5)

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
    camera_metrics = manager.metrics()
    detector_metrics = detector.metrics() if detector else None
    detector_ready = bool(detector_metrics and detector_metrics.get("ready"))
    return {
        "status": "ok" if detector_ready or detector is None else "degraded",
        "mode": _mode(),
        "profile": str(core_cfg.get("profile", "detection-tracking-v1")),
        "cameras": camera_metrics,
        "online": sum(bool(value.get("online")) for value in camera_metrics.values()),
        "total": len(camera_metrics),
        "detector": detector_metrics,
        "reid": reid.metrics() if reid else {"enabled": False, "ready": False},
        "face": face.metrics() if face else {"enabled": False, "ready": False},
        "publishers": {cid: publisher.metrics() for cid, publisher in publishers.items()},
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


@app.get("/tracks")
def tracks():
    cameras_payload = {}
    total = 0
    for camera_id, publisher in publishers.items():
        items = publisher.track_snapshot()
        enriched = []
        for item in items:
            row = dict(item)
            if identity_provider is not None:
                identity = identity_provider.identity_for_track(
                    camera_id,
                    int(row.get("track_id") or 0),
                )
                if identity:
                    for key in (
                        "global_id",
                        "reid_similarity",
                        "reid_reason",
                        "reid_provisional",
                        "known",
                        "name",
                        "person_id",
                        "known_id",
                        "department",
                        "face_similarity",
                        "face_reason",
                    ):
                        if key in identity:
                            row[key] = identity.get(key)
            enriched.append(row)
        cameras_payload[camera_id] = {
            "count": len(enriched),
            "tracks": enriched,
            "metrics": publisher.visual_tracker.metrics(),
        }
        total += len(enriched)
    return {
        "enabled": True,
        "scope": "camera_local+room_consensus_reid+face" if face else (
            "camera_local+room_consensus_reid" if reid else "camera_local"
        ),
        "cross_camera_identity": reid is not None,
        "known_identity": face is not None,
        "total": total,
        "cameras": cameras_payload,
    }


@app.get("/reid")
def reid_status():
    if reid is None:
        return {"enabled": False, "ready": False, "identities": [], "bindings": []}
    metrics = reid.metrics()
    payload = reid.snapshot()
    return {
        "enabled": True,
        "ready": bool(metrics.get("ready")),
        "metrics": metrics,
        **payload,
    }


@app.get("/faces")
def faces_status():
    if face is None:
        return {"enabled": False, "ready": False, "people": [], "bindings": []}
    payload = face.snapshot()
    return {"enabled": True, "ready": bool(payload["metrics"].get("ready")), **payload}


@app.post("/faces/enrollment/sample/{camera_id}/{track_id}")
def face_enrollment_sample(camera_id: str, track_id: int):
    if face is None:
        raise HTTPException(503, "face recognition is disabled")
    try:
        return face.capture_enrollment_sample(camera_id, track_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/faces/enrollment/commit")
def face_enrollment_commit(request: FaceEnrollRequest):
    if face is None:
        raise HTTPException(503, "face recognition is disabled")
    try:
        return face.commit_enrollment(
            request.name,
            request.department,
            request.employee_id,
            request.sample_tokens,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/faces/avatar/{person_id}")
def face_avatar(person_id: str):
    if face is None:
        raise HTTPException(404, "face recognition is disabled")
    payload = face.avatar(person_id)
    if not payload:
        raise HTTPException(404, "avatar not found")
    return Response(content=payload, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.delete("/faces/people/{person_id}")
def face_delete_person(person_id: str):
    if face is None:
        raise HTTPException(404, "face recognition is disabled")
    if not face.delete_person(person_id):
        raise HTTPException(404, "person not found")
    return {"deleted": True, "person_id": person_id}


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
        after, wait_ms / 1000.0
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
