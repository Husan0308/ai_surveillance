"""YOLO26n-pose detector backend for Camera V2.

Detection only:
- YOLO26n-pose
- imgsz=832
- low confidence candidate threshold
- pose/keypoint validation
- duplicate suppression

The worker preserves the existing CameraDetectionV2 job/result contract.
"""

from __future__ import annotations

import importlib.metadata
import os
import time
from pathlib import Path

import numpy as np


DEFAULT_IMGSZ = 832
DEFAULT_CONF = 0.10
DEFAULT_IOU = 0.80


def _overlap(a, b) -> tuple[float, float]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    bb = max(1.0, (bx2 - bx1) * (by2 - by1))

    union = aa + bb - inter

    iou = inter / max(union, 1e-6)
    containment = inter / max(min(aa, bb), 1e-6)

    return iou, containment


def _rows_from_result(result, max_det: int = 300):
    """Convert one Ultralytics pose Result to CameraDetectionV2 rows."""

    if result.boxes is None or result.keypoints is None:
        return []

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    kpts = result.keypoints.data.detach().cpu().numpy()

    candidates = []

    for box, conf, kp in zip(boxes, confs, kpts):
        if kp.ndim != 2 or kp.shape[1] < 3:
            continue

        kp_conf = kp[:, 2]

        strong = int((kp_conf >= 0.50).sum())
        usable = int((kp_conf >= 0.25).sum())
        conf = float(conf)

        # Preserve seated / partially occluded people.
        # Very low-confidence candidates require stronger pose evidence.
        if conf < 0.20:
            if usable < 4 or strong < 2:
                continue
        else:
            if usable < 2:
                continue

        x1, y1, x2, y2 = map(float, box)

        quality = (
            conf
            + 0.025 * usable
            + 0.015 * strong
        )

        valid_kp = kp_conf >= 0.25

        candidates.append(
            {
                "box": (x1, y1, x2, y2),
                "conf": conf,
                "strong": strong,
                "usable": usable,
                "quality": quality,
                "kp_xy": kp[:, :2].copy(),
                "kp_conf": kp_conf.copy(),
                "valid_kp": valid_kp.copy(),
            }
        )

    # Same detection-only dedupe that worked in the CAM-01 live test.
    candidates.sort(key=lambda x: x["quality"], reverse=True)

    kept = []

    for cand in candidates:
        duplicate = False

        for old in kept:
            iou, containment = _overlap(
                cand["box"],
                old["box"],
            )

            # Do NOT remove detections merely because person boxes overlap.
            # Two real people can heavily overlap during occlusion.
            common = cand["valid_kp"] & old["valid_kp"]

            same_pose = False

            if int(common.sum()) >= 3:
                a = cand["kp_xy"][common]
                b = old["kp_xy"][common]

                distances = np.linalg.norm(a - b, axis=1)

                ax1, ay1, ax2, ay2 = cand["box"]
                bx1, by1, bx2, by2 = old["box"]

                diag_a = np.hypot(ax2 - ax1, ay2 - ay1)
                diag_b = np.hypot(bx2 - bx1, by2 - by1)

                scale = max(1.0, min(diag_a, diag_b))

                pose_distance = float(
                    np.median(distances) / scale
                )

                # Duplicate predictions of the SAME person have almost
                # identical keypoints. Different overlapping people don't.
                same_pose = pose_distance <= 0.10

            if same_pose and (
                iou >= 0.35 or containment >= 0.65
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(cand)

        if len(kept) >= max_det:
            break

    return [
        (cand["box"], cand["conf"])
        for cand in kept
    ]


def yolo_pose_worker(job_q, result_q) -> None:
    """Spawn-safe CUDA YOLO26n-pose worker."""

    try:
        try:
            os.nice(8)
        except Exception:
            pass

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")

        torch.cuda.set_device(0)

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        torch.backends.cudnn.benchmark = True

        from . import detection as det

        startup_delay = float(
            os.environ.get(
                "CAMERA_V2_DETECT_STARTUP_DELAY",
                "3.0",
            )
        )

        if startup_delay > 0:
            time.sleep(startup_delay)

        imgsz = int(
            os.environ.get(
                "CAMERA_V2_POSE_IMGSZ",
                str(DEFAULT_IMGSZ),
            )
        )

        conf = float(
            os.environ.get(
                "CAMERA_V2_POSE_CONF",
                str(DEFAULT_CONF),
            )
        )

        iou = float(
            os.environ.get(
                "CAMERA_V2_POSE_IOU",
                str(DEFAULT_IOU),
            )
        )

        max_det = int(det.MAX_DET)

        default_model = (
            Path(__file__).resolve().parents[2]
            / "yolo26s-pose.pt"
        )

        model_path = Path(
            os.environ.get(
                "CAMERA_V2_POSE_MODEL",
                str(default_model),
            )
        )

        if not model_path.exists():
            raise RuntimeError(
                f"pose model not found: {model_path}"
            )

        model = YOLO(str(model_path))

        # Warmup using the same 1280x720 BGR analysis-frame geometry.
        warm = np.zeros(
            (720, 1280, 3),
            dtype=np.uint8,
        )

        model.predict(
            warm,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            classes=[0],
            max_det=max_det,
            device=0,
            verbose=False,
        )

        try:
            version = importlib.metadata.version("ultralytics")
        except Exception:
            version = "unknown"

        result_q.put(
            {
                "type": "ready",
                "backend": "YOLO26n-pose",
                "device": torch.cuda.get_device_name(0),
                "cuda": str(torch.version.cuda),
                "model": str(model_path),
                "version": version,
                "imgsz": imgsz,
                "threshold": conf,
            }
        )

        while True:
            job = job_q.get()

            if job is None:
                return

            started = time.monotonic()

            try:
                # GStreamer analysis frames are already BGR NumPy arrays.
                predictions = model.predict(
                    job["frames"],
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    classes=[0],
                    max_det=max_det,
                    device=0,
                    verbose=False,
                )

                ended = time.monotonic()

                if not isinstance(predictions, (list, tuple)):
                    predictions = [predictions]

                if len(predictions) != len(job["cameras"]):
                    raise RuntimeError(
                        "YOLO pose batch mismatch: "
                        f"predictions={len(predictions)} "
                        f"cameras={len(job['cameras'])}"
                    )

                output = {}

                for cid, prediction in zip(
                    job["cameras"],
                    predictions,
                ):
                    output[cid] = _rows_from_result(
                        prediction,
                        max_det=max_det,
                    )

                result_q.put(
                    {
                        "type": "result",
                        "backend": "YOLO26n-pose",
                        "cameras": job["cameras"],
                        "captured": job["captured"],
                        "boxes": output,
                        "fragment_rejected": {
                            cid: 0
                            for cid in job["cameras"]
                        },
                        "batch_ms": (
                            ended - started
                        ) * 1000.0,
                    }
                )

            except torch.cuda.OutOfMemoryError as exc:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                result_q.put(
                    {
                        "type": "batch_error",
                        "error": (
                            "YOLO26n-pose CUDA OOM: "
                            f"{exc}"
                        ),
                    }
                )

            except BaseException as exc:
                result_q.put(
                    {
                        "type": "batch_error",
                        "error": (
                            "YOLO26n-pose "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )

    except BaseException as exc:
        result_q.put(
            {
                "type": "fatal",
                "error": (
                    "YOLO26n-pose "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )


def _capture_gate_until_sample(self, _pad, _info, cid: str):
    """Pass buffers only while a detector capture is armed."""

    with self.capture_lock:
        requested = bool(
            self.capture_requested.get(cid, False)
        )

    if not requested:
        return self.Gst.PadProbeReturn.DROP

    return self.Gst.PadProbeReturn.OK


def install() -> None:
    """Install YOLO26n-pose into CameraDetectionV2."""

    from . import detection

    detection._yolo_worker = yolo_pose_worker
    detection.CameraDetectionV2._infer_gate_probe = (
        _capture_gate_until_sample
    )

    pascal_safe = os.environ.get(
        "CAMERA_V2_PASCAL_SAFE",
        "0",
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if not pascal_safe:
        from .sparse_tracker_contract import (
            install_sparse_tracker_contract,
        )

        install_sparse_tracker_contract()
