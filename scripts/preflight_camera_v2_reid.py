from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def main() -> int:
    backend = reid_backend()
    print(f"REID_PREFLIGHT backend={backend}")
    if backend != "external":
        raise RuntimeError(
            "This GTX 1050 Ti preflight expects CAMERA_V2_REID_BACKEND=external"
        )

    model = resolve_reid_model()
    print(f"REID_PREFLIGHT model={model} size={model.stat().st_size / (1024 * 1024):.1f}MiB")

    import cv2

    print(f"REID_PREFLIGHT opencv={cv2.__version__}")
    net = cv2.dnn.readNetFromONNX(str(model))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    net.setInput(np.zeros((1, 3, 256, 128), dtype=np.float32))
    output = np.asarray(net.forward(), dtype=np.float32).reshape(-1)
    if output.size != 256:
        raise RuntimeError(f"unexpected ONNX ReID output size: {output.size}, expected 256")
    if not np.all(np.isfinite(output)):
        raise RuntimeError("ONNX ReID output contains NaN/Inf")
    print("REID_PREFLIGHT onnx_forward=PASS feature_size=256")

    generated = prepare_sparse_tracker_config(_deepstream_config())
    text = generated.read_text(encoding="utf-8")
    required = ("enableReAssoc: 0", "reidType: 0", "outputReidTensor: 0")
    for item in required:
        if item not in text:
            raise RuntimeError(f"generated NvDCF config did not disable TensorRT ReID: missing {item}")
    forbidden = ("modelEngineFile:", "onnxFile:", "tltEncodedModel:")
    for item in forbidden:
        if item in text:
            raise RuntimeError(f"generated NvDCF config still references a TensorRT ReID model: {item}")
    print(f"REID_PREFLIGHT nvdcf_config=PASS path={generated}")
    print("CAMERA_V2_REID_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())