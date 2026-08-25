from __future__ import annotations

"""ONNX Runtime CPU detector backend for Camera V2.

The production profile uses the exported YOLO26s-pose ONNX model on CPU so the
GTX 1050 Ti is left to NVDEC/display/ReID work.  The worker intentionally keeps
the same pose/keypoint validation and overlap-safe de-duplication that was used
by the earlier CAM-01 CUDA pose detector.
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


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def yolo_onnx_cpu_worker(job_q, result_q) -> None:
    try:
        try:
            # Keep ingest/display callbacks ahead of CPU inference under load.
            os.nice(6)
        except Exception:
            pass

        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        import cv2
        from ultralytics import YOLO

        cv2.setNumThreads(1)

        from . import detection as det
        from .yolo_pose_backend import _rows_from_result

        startup_delay = float(
            os.environ.get("CAMERA_V2_DETECT_STARTUP_DELAY", "0.5")
        )
        if startup_delay > 0:
            time.sleep(startup_delay)

        model_path = _resolve_model(
            os.environ.get("CAMERA_V2_YOLO_MODEL", "yolo26s-pose.onnx")
        )
        if not model_path.lower().endswith(".onnx"):
            raise RuntimeError(
                "ONNX CPU backend requires a .onnx model, "
                f"got {model_path!r}"
            )

        task = os.environ.get("CAMERA_V2_DETECT_TASK", "pose").strip().lower()
        if task not in {"detect", "pose"}:
            raise RuntimeError(
                "CAMERA_V2_DETECT_TASK must be 'detect' or 'pose', "
                f"got {task!r}"
            )

        if task == "pose":
            input_width = max(
                32,
                int(os.environ.get("CAMERA_V2_POSE_INPUT_WIDTH", "832")),
            )
            input_height = max(
                32,
                int(os.environ.get("CAMERA_V2_POSE_INPUT_HEIGHT", "480")),
            )
            conf = float(os.environ.get("CAMERA_V2_POSE_CONF", "0.10"))
            iou = float(os.environ.get("CAMERA_V2_POSE_IOU", "0.80"))
            inference_size = (input_height, input_width)
        else:
            conf = float(det.CONF)
            iou = float(det.IOU)
            inference_size = (det.INFER_HEIGHT, det.INFER_WIDTH)

        model = YOLO(model_path, task=task)
        kwargs = {
            "imgsz": inference_size,
            "classes": [0],
            "conf": conf,
            "iou": iou,
            "max_det": int(det.MAX_DET),
            "device": "cpu",
            "verbose": False,
            "stream": False,
        }

        # The live pose path intentionally receives a high-resolution source
        # crop and lets Ultralytics/ONNX resize once to the fixed 832x480 model
        # tensor.  This preserves small/occluded people better than first
        # shrinking the analysis branch to 672x384.
        warm_width = max(
            inference_size[1],
            int(os.environ.get("CAMERA_V2_ANALYSIS_TILE_WIDTH", "1280")),
        )
        warm_height = max(
            inference_size[0],
            int(os.environ.get("CAMERA_V2_ANALYSIS_TILE_HEIGHT", "720")),
        )
        warm = [
            np.zeros((warm_height, warm_width, 3), dtype=np.uint8)
            for _ in range(det.MICRO_BATCH)
        ]
        model.predict(source=warm, **kwargs)

        result_q.put(
            {
                "type": "ready",
                "device": "CPUExecutionProvider",
                "cuda": "none",
                "model": model_path,
                "backend": (
                    "onnxruntime-cpu-pose"
                    if task == "pose"
                    else "onnxruntime-cpu-detect"
                ),
                "task": task,
                "imgsz": list(inference_size),
                "threshold": conf,
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

                if not isinstance(predictions, (list, tuple)):
                    predictions = [predictions]
                if len(predictions) != len(job["cameras"]):
                    raise RuntimeError(
                        "ONNX result batch mismatch: "
                        f"predictions={len(predictions)} "
                        f"cameras={len(job['cameras'])}"
                    )

                output = {}
                if task == "pose":
                    for cid, prediction in zip(job["cameras"], predictions):
                        output[cid] = _rows_from_result(
                            prediction,
                            max_det=int(det.MAX_DET),
                        )
                else:
                    for cid, prediction in zip(job["cameras"], predictions):
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
                        "backend": (
                            "onnxruntime-cpu-pose"
                            if task == "pose"
                            else "onnxruntime-cpu-detect"
                        ),
                        "task": task,
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
                            f"YOLO26s {task} ONNX CPU "
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
