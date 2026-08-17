from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.external_reid import ExternalReIDWorker
from services.camera_v2.global_reid import GlobalReIDManager
from services.camera_v2.person_tracking_final import CameraPersonTrackingFinal
from services.camera_v2.tracker_profile import (
    prepare_sparse_tracker_config,
    reid_backend,
    resolve_reid_model,
)


def _deepstream_config() -> Path:
    candidates = [
        Path("/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml")
    ]
    candidates.extend(
        sorted(
            Path("/opt/nvidia/deepstream").glob(
                "deepstream-*/samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml"
            ),
            reverse=True,
        )
    )
    for path in candidates:
        if path.exists():
            return path
    raise RuntimeError("DeepStream NvDCF max-perf config was not found")


def _normalize(raw: np.ndarray) -> np.ndarray:
    vector = np.asarray(raw, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size != 256 or not np.isfinite(norm) or norm <= 1e-8:
        raise RuntimeError(f"invalid ReID feature: size={vector.size} norm={norm}")
    return vector / norm


def _validate_global_identity_logic() -> None:
    manager = GlobalReIDManager()
    expected_rooms = {0: 0, 1: 0, 2: 1, 5: 1, 4: 2, 3: 2}
    actual_rooms = {sid: manager.room_of(sid) for sid in expected_rooms}
    if actual_rooms != expected_rooms:
        raise RuntimeError(
            f"wrong Camera V2 ReID room topology: got={actual_rooms} expected={expected_rooms}"
        )

    vector = np.zeros(256, dtype=np.float32)
    vector[0] = 1.0
    feature = tuple(float(v) for v in vector)
    color = np.zeros(96, dtype=np.float32)
    color[3] = 1.0
    color_feature = tuple(float(v) for v in color)

    row_a = {
        "source_id": 2,
        "object_id": 101,
        "feature": feature,
        "color_feature": color_feature,
        "bbox": (100.0, 100.0, 180.0, 300.0),
        "confidence": 0.90,
        "tracker_confidence": 0.70,
    }
    row_b = {
        "source_id": 5,
        "object_id": 202,
        "feature": feature,
        "color_feature": color_feature,
        "bbox": (400.0, 90.0, 480.0, 300.0),
        "confidence": 0.90,
        "tracker_confidence": 0.70,
    }
    manager.update_active_tracks([row_a], now=100.0)
    manager.observe([row_a], now=100.0)
    manager.update_active_tracks([row_b], now=100.2)
    manager.observe([row_b], now=100.2)

    labels = {(sid, oid): label for sid, oid, label in manager.label_assignments()}
    if labels.get((2, 101)) != labels.get((5, 202)):
        raise RuntimeError(f"same-room cross-camera ReID did not preserve one global ID: {labels}")

    # Simultaneous different-room appearance must not teleport into the active ID.
    row_c = {
        "source_id": 3,
        "object_id": 303,
        "feature": feature,
        "color_feature": color_feature,
        "bbox": (200.0, 80.0, 280.0, 300.0),
        "confidence": 0.90,
        "tracker_confidence": 0.70,
    }
    manager.update_active_tracks([row_c], now=100.3)
    manager.observe([row_c], now=100.3)
    labels = {(sid, oid): label for sid, oid, label in manager.label_assignments()}
    if labels.get((3, 303)) == labels.get((2, 101)):
        raise RuntimeError("cross-room active conflict guard failed")

    # Same-camera NvDCF ID reset after the old track is no longer active should
    # preserve the global identity using appearance + spatial continuity.
    manager2 = GlobalReIDManager()
    first = {
        "source_id": 0,
        "object_id": 11,
        "feature": feature,
        "color_feature": color_feature,
        "bbox": (120.0, 100.0, 210.0, 330.0),
        "confidence": 0.90,
        "tracker_confidence": 0.65,
    }
    manager2.update_active_tracks([first], now=200.0)
    manager2.observe([first], now=200.0)
    second = dict(first)
    second["object_id"] = 12
    second["bbox"] = (132.0, 102.0, 222.0, 332.0)
    manager2.update_active_tracks([second], now=201.0)
    manager2.observe([second], now=201.0)
    labels2 = {(sid, oid): label for sid, oid, label in manager2.label_assignments()}
    if labels2.get((0, 11)) != labels2.get((0, 12)):
        raise RuntimeError(f"same-camera ID-reset continuity failed: {labels2}")

    snapshot = manager.snapshot()
    snapshot2 = manager2.snapshot()
    if snapshot["stats"].get("direct_match", 0) < 1:
        raise RuntimeError("global ReID direct-match path was not exercised")
    if snapshot2["stats"].get("continuation_match", 0) < 1:
        raise RuntimeError("same-camera continuation path was not exercised")
    print(
        "REID_PREFLIGHT global_identity=PASS "
        f"room_map={snapshot['room_map']} direct={snapshot['stats']['direct_match']} "
        f"continuation={snapshot2['stats']['continuation_match']}"
    )


def main() -> int:
    backend = reid_backend()
    print(f"REID_PREFLIGHT backend={backend}")
    if backend != "external":
        raise RuntimeError("This GTX 1050 Ti preflight expects CAMERA_V2_REID_BACKEND=external")

    model = resolve_reid_model()
    print(f"REID_PREFLIGHT model={model} size={model.stat().st_size / (1024 * 1024):.1f}MiB")

    import cv2

    print(f"REID_PREFLIGHT opencv={cv2.__version__}")
    net = cv2.dnn.readNetFromONNX(str(model))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    test_crop = np.zeros((300, 92, 3), dtype=np.uint8)
    test_crop[:, :46, 1] = 180
    test_crop[:, 46:, 2] = 160
    blob = ExternalReIDWorker._blob(test_crop)
    if blob.shape != (1, 3, 256, 128):
        raise RuntimeError(f"unexpected production ReID blob shape: {blob.shape}")
    if not np.all(np.isfinite(blob)):
        raise RuntimeError("production ReID blob contains NaN/Inf")
    color = ExternalReIDWorker._color_signature(test_crop)
    if len(color) != ExternalReIDWorker.COLOR_FEATURE_SIZE:
        raise RuntimeError(f"unexpected production colour feature size: {len(color)}")

    net.setInput(blob)
    feature_a = _normalize(net.forward())
    net.setInput(blob.copy())
    feature_b = _normalize(net.forward())
    cosine = float(np.dot(feature_a, feature_b))
    if cosine < 0.999:
        raise RuntimeError(f"ReID repeatability check failed: cosine={cosine:.6f}")
    print(
        "REID_PREFLIGHT onnx_forward=PASS "
        f"feature_size=256 color_size={len(color)} repeat_cosine={cosine:.6f} "
        "preprocess=tao-direct-resize"
    )

    _validate_global_identity_logic()

    generated = prepare_sparse_tracker_config(_deepstream_config())
    generated = CameraPersonTrackingFinal._stabilize_tracker_config(generated)
    text = generated.read_text(encoding="utf-8")
    required = (
        "enableReAssoc: 0",
        "reidType: 0",
        "outputReidTensor: 0",
        "enableBboxUnClipping: 0",
        "minIouDiff4NewTarget: 0.60",
        "probationAge: 2",
        "maxShadowTrackingAge: 50",
        "minTrackingConfidenceDuringInactive: 0.22",
    )
    for item in required:
        if item not in text:
            raise RuntimeError(f"generated NvDCF config missing runtime requirement: {item}")

    forbidden = ("modelEngineFile:", "onnxFile:", "tltEncodedModel:")
    for item in forbidden:
        if item in text:
            raise RuntimeError(f"generated NvDCF config still references TensorRT ReID: {item}")

    print(f"REID_PREFLIGHT nvdcf_config=PASS path={generated}")
    print("CAMERA_V2_REID_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
