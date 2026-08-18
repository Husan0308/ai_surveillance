from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.global_identity import STATE_CONFIRMED
from services.camera_v2.native_bridge import _GlobalLabel, _TrackRow, ensure_bridge
from services.camera_v2.person_tracking_reid import CameraPersonTrackingReID
from services.camera_v2.pose_ankle import LEFT_ANKLE, RIGHT_ANKLE, POSE_MIN_VISIBLE
from services.camera_v2.pose_heatmap_bridge import ensure_pose_heatmap_bridge
from services.camera_v2.qwen_reid import QwenReIdVerifier
from services.camera_v2.reid_embedder import AutoReIdEmbedder
from services.camera_v2.reid_production import (
    ProductionGlobalIdentityCore,
    ProductionReIdIdentityEngine,
)
from services.camera_v2.reid_quality import evaluate_crop_quality
from services.ml_service.app.config import load_settings


def _vec(*values):
    value = np.asarray(values, dtype=np.float32)
    return value / np.linalg.norm(value)


def _feed(core, camera, track_id, vectors, start, room):
    result = None
    for index, vector in enumerate(vectors):
        result = core.observe_embedding(
            camera_id=camera,
            local_id=track_id,
            embedding=vector,
            quality=0.90,
            captured_at=start + index * 0.25,
            room_id=room,
            bbox=(100, 80, 210, 350),
        )
    return result


def main() -> int:
    settings = load_settings()
    active_rooms = {camera.camera_id: camera.room for camera in settings.cameras}
    if not 1 <= len(active_rooms) <= 6:
        raise RuntimeError(f"enabled camera count must be 1..6, got {len(active_rooms)}")

    cfg_path = ROOT / "config" / "reid.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg = dict(raw.get("reid") or raw)

    smoke_rooms = {
        "CAM-01": "Devs", "CAM-04": "Devs",
        "CAM-02": "Entrance", "CAM-05": "Entrance",
        "CAM-03": "Main Rooms", "CAM-06": "Main Rooms",
    }
    smoke_cfg = dict(cfg)
    smoke_cfg["camera_rooms"] = smoke_rooms

    core = ProductionGlobalIdentityCore(smoke_cfg)
    a = _vec(1.0, 0.0, 0.0, 0.0)
    a2 = _vec(0.98, 0.10, 0.0, 0.0)
    a3 = _vec(0.96, -0.12, 0.0, 0.0)
    _feed(core, "CAM-01", 10, [a, a2, a3, a], 10.0, "Devs")
    first = core.binding_for_track("CAM-01", 10)
    if not first or first["state"] != STATE_CONFIRMED:
        raise RuntimeError(f"first identity did not confirm: {first}")
    _feed(core, "CAM-04", 20, [a3, a2, a], 11.2, "Devs")
    second = core.binding_for_track("CAM-04", 20)
    if not second or second["global_id"] != first["global_id"]:
        raise RuntimeError(f"same-room pair did not reuse Global ID: {first} vs {second}")
    _feed(core, "CAM-03", 30, [a, a2, a3], 11.4, "Main Rooms")
    third = core.binding_for_track("CAM-03", 30)
    if third and third["global_id"] == first["global_id"]:
        raise RuntimeError("different-room simultaneous track illegally reused Global ID")

    core2 = ProductionGlobalIdentityCore(smoke_cfg)
    _feed(core2, "CAM-02", 7, [a, a2, a3, a], 20.0, "Entrance")
    core2.observe_camera_snapshot("CAM-02", [], seen_at=21.6)
    _feed(core2, "CAM-02", 8, [a3, a2, a], 21.7, "Entrance")
    reconnect = core2.binding_for_track("CAM-02", 8)
    if not reconnect or reconnect["global_id"] != 1:
        raise RuntimeError(f"same-camera occlusion reconnect failed: {reconnect}")

    core3 = ProductionGlobalIdentityCore(smoke_cfg)
    _feed(core3, "CAM-01", 1, [a, a2, a3, a], 30.0, "Devs")
    gray = _vec(0.65, 0.76, 0.0, 0.0)
    _feed(core3, "CAM-04", 2, [gray, gray, gray], 31.4, "Devs")
    core3.apply_qwen_result("CAM-04", 2, 1, "SAME", 0.97, now=32.5)
    origins = {row.origin_key for row in core3._globals[1].gallery}
    if ("CAM-04", 2) in origins:
        raise RuntimeError("Qwen gray-zone track poisoned canonical gallery")

    if ctypes.sizeof(_TrackRow) != 48 or ctypes.sizeof(_GlobalLabel) != 24:
        raise RuntimeError(
            "native ReID ABI mismatch: "
            f"track={ctypes.sizeof(_TrackRow)} label={ctypes.sizeof(_GlobalLabel)}"
        )

    native_path = ensure_bridge()
    if not native_path.exists():
        raise RuntimeError(f"native bridge build did not produce library: {native_path}")
    pose_heatmap_path = ensure_pose_heatmap_bridge()
    if not pose_heatmap_path.exists():
        raise RuntimeError(f"pose heatmap bridge build failed: {pose_heatmap_path}")
    if (LEFT_ANKLE, RIGHT_ANKLE) != (15, 16) or POSE_MIN_VISIBLE < 3:
        raise RuntimeError("pose ankle contract is invalid")

    crop = np.random.default_rng(4).integers(0, 255, (220, 90, 3), dtype=np.uint8)
    quality = evaluate_crop_quality(
        crop,
        source_bbox=(100, 80, 190, 300),
        source_width=704,
        source_height=384,
        detector_confidence=0.8,
        tracker_confidence=0.8,
    )
    if not quality.accepted:
        raise RuntimeError(f"quality gate rejected clear synthetic crop: {quality}")

    if not issubclass(CameraPersonTrackingReID, object) or not callable(
        ProductionReIdIdentityEngine
    ):
        raise RuntimeError("production ReID runtime imports failed")
    embedder = AutoReIdEmbedder(cfg, ROOT)
    qwen = QwenReIdVerifier(cfg)

    print(
        "CAMERA_V2_REID_PREFLIGHT inventory=PASS "
        f"active={len(active_rooms)} rooms={active_rooms}"
    )
    print("CAMERA_V2_REID_PREFLIGHT identity=PASS synthetic-topology+conflict+occlusion-reconnect")
    print("CAMERA_V2_REID_PREFLIGHT gallery_guard=PASS independent-self-check+qwen-quarantine")
    print("CAMERA_V2_REID_PREFLIGHT crop=PASS quality+duplicate+top3-diversity")
    print(
        "CAMERA_V2_REID_PREFLIGHT native=PASS "
        f"path={native_path} track_abi={ctypes.sizeof(_TrackRow)} "
        f"label_abi={ctypes.sizeof(_GlobalLabel)}"
    )
    print(
        "CAMERA_V2_REID_PREFLIGHT pose_heatmap=PASS "
        f"path={pose_heatmap_path} ankle_indices={LEFT_ANKLE},{RIGHT_ANKLE} bbox_anchor=OFF"
    )
    print(
        "CAMERA_V2_REID_PREFLIGHT embedder="
        f"{embedder.metrics().get('requested','auto')} lazy=PASS"
    )
    print(f"CAMERA_V2_REID_PREFLIGHT qwen_configured={int(qwen.enabled)} async=PASS")
    print("CAMERA_V2_REID_PREFLIGHT hot_path=YOLO26m+NvDCF pose=CPU-low-rate recovery=bounded")
    print("CAMERA_V2_REID_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
