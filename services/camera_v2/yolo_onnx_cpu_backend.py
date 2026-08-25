from __future__ import annotations

"""ONNX Runtime CPU detector backend for Camera V2.

This backend preserves the CameraDetectionV2 multiprocessing job/result contract
while keeping person detection off the GTX 1050 Ti. It is intended for the
CAM-01 tuning profile where the six-camera wall remains GPU-decoded/rendered and
only one camera is sampled for AI.
"""

import os
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _resolve_model(spec: str) -> str:
    path = Path(spec)
    if path.is_file():
        return str(path)
    path = ROOT / spec
    return str(path) if path.is_file() else spec


def yolo_onnx_cpu_worker(job_q, result_q) -> None:
    try:
        try:
            # Camera ingest/display remain more important than detector CPU work.
            os.nice(6)
        except Exception:
            pass

        # Avoid OpenCV/BLAS thread explosions around ONNX Runtime. ONNX Runtime
        # keeps its own optimized CPU thread pool.
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        import cv2
        from ultralytics import YOLO

        cv2.setNumThreads(1)

        from . import detection as det

        startup_delay = float(
            os.environ.get("CAMERA_V2_DETECT_STARTUP_DELAY", "0.5")
        )
        if startup_delay > 0:
            time.sleep(startup_delay)

        model_path = _resolve_model(
            os.environ.get("CAMERA_V2_YOLO_MODEL", "yolo26s.onnx")
        )
        if not model_path.lower().endswith(".onnx"):
            raise RuntimeError(
                "ONNX CPU backend requires a .onnx model, "
                f"got {model_path!r}"
            )

        model = YOLO(model_path, task="detect")
        kwargs = {
            "imgsz": (det.INFER_HEIGHT, det.INFER_WIDTH),
            "rect": True,
            "classes": [0],
            "conf": float(det.CONF),
            "iou": float(det.IOU),
            "max_det": int(det.MAX_DET),
            "device": "cpu",
            "verbose": False,
            "stream": False,
        }

        warm = [
            np.zeros(
                (det.INFER_HEIGHT, det.INFER_WIDTH, 3),
                dtype=np.uint8,
            )
            for _ in range(det.MICRO_BATCH)
        ]
        model.predict(source=warm, **kwargs)

        result_q.put(
            {
                "type": "ready",
                "device": "CPUExecutionProvider",
                "cuda": "none",
                "model": model_path,
                "backend": "onnxruntime-cpu",
            }
        )

        while True:
            job = job_q.get()
            if job is None:
                return

            started = time.monotonic()
            try:
                predictions = model.predict(
                    source=job["frames"],
                    **kwargs,
                )
                ended = time.monotonic()

                output = {}
                for cid, prediction in zip(
                    job["cameras"], predictions
                ):
                    boxes = getattr(prediction, "boxes", None)
                    rows = []
                    if boxes is not None and len(boxes):
                        xyxy = boxes.xyxy.detach().cpu().tolist()
                        confs = boxes.conf.detach().cpu().tolist()
                        for coords, score in zip(xyxy, confs):
                            rows.append(
                                (
                                    [float(v) for v in coords],
                                    float(score),
                                )
                            )
                    output[cid] = rows

                result_q.put(
                    {
                        "type": "result",
                        "backend": "onnxruntime-cpu",
                        "cameras": job["cameras"],
                        "captured": job["captured"],
                        "boxes": output,
                        "batch_ms": (ended - started) * 1000.0,
                    }
                )
            except BaseException as exc:
                result_q.put(
                    {
                        "type": "batch_error",
                        "error": (
                            "YOLO26s ONNX CPU "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )

    except BaseException as exc:
        result_q.put(
            {
                "type": "fatal",
                "error": (
                    "YOLO26s ONNX CPU "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )


def install() -> None:
    """Install the ONNX CPU worker into CameraDetectionV2."""

    from . import detection

    detection._yolo_worker = yolo_onnx_cpu_worker
