from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.external_reid import ExternalReIDWorker
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
        raise RuntimeError(
            f"invalid ReID feature: size={vector.size} norm={norm}"
        )
    return vector / norm


def main() -> int:
    backend = reid_backend()
    print(f"REID_PREFLIGHT backend={backend}")
    if backend != "external":
        raise RuntimeError(
            "This GTX 1050 Ti preflight expects CAMERA_V2_REID_BACKEND=external"
        )

    model = resolve_reid_model()
    print(
        f"REID_PREFLIGHT model={model} "
        f"size={model.stat().st_size / (1024 * 1024):.1f}MiB"
    )

    import cv2

    print(f"REID_PREFLIGHT opencv={cv2.__version__}")
    net = cv2.dnn.readNetFromONNX(str(model))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Exercise the exact production preprocessing path, not an arbitrary zero blob.
    test_crop = np.zeros((300, 92, 3), dtype=np.uint8)
    test_crop[:, :46, 1] = 180
    test_crop[:, 46:, 2] = 160
    blob = ExternalReIDWorker._blob(test_crop)
    if blob.shape != (1, 3, 256, 128):
        raise RuntimeError(f"unexpected production ReID blob shape: {blob.shape}")
    if not np.all(np.isfinite(blob)):
        raise RuntimeError("production ReID blob contains NaN/Inf")

    net.setInput(blob)
    feature_a = _normalize(net.forward())
    net.setInput(blob.copy())
    feature_b = _normalize(net.forward())
    cosine = float(np.dot(feature_a, feature_b))
    if cosine < 0.999:
        raise RuntimeError(
            f"ReID repeatability check failed: cosine={cosine:.6f}"
        )
    print(
        "REID_PREFLIGHT onnx_forward=PASS "
        f"feature_size=256 repeat_cosine={cosine:.6f} preprocess=tao-direct-resize"
    )

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
        "maxShadowTrackingAge: 38",
    )
    for item in required:
        if item not in text:
            raise RuntimeError(
                f"generated NvDCF config missing runtime requirement: {item}"
            )

    forbidden = ("modelEngineFile:", "onnxFile:", "tltEncodedModel:")
    for item in forbidden:
        if item in text:
            raise RuntimeError(
                f"generated NvDCF config still references TensorRT ReID: {item}"
            )

    print(f"REID_PREFLIGHT nvdcf_config=PASS path={generated}")
    print("CAMERA_V2_REID_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
