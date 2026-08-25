"""YOLO26s-pose detector backend for Camera V2 sticky tracking.

Pose is used only to validate person detections. NvDCF remains the temporal
tracker. The validator is intentionally tolerant of seated, bent and partially
occluded people while still requiring body-keypoint evidence for weak boxes.
"""

from __future__ import annotations

import importlib.metadata
import os
import time
from pathlib import Path

import numpy as np

DEFAULT_IMGSZ = 832
DEFAULT_CONF = 0.05
DEFAULT_IOU = 0.80
_DIAG_CALLS = 0


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
    return inter / max(union, 1e-6), inter / max(min(aa, bb), 1e-6)


def _rows_from_result(result, max_det: int = 300):
    """Convert one Ultralytics pose Result to validated person bbox rows."""
    global _DIAG_CALLS
    _DIAG_CALLS += 1

    if result.boxes is None or result.keypoints is None:
        if _DIAG_CALLS <= 3 or _DIAG_CALLS % 10 == 0:
            print(
                f"CAMERA_POSE_DIAG n={_DIAG_CALLS} raw=0 kept=0 reason=no_boxes_or_keypoints",
                flush=True,
            )
        return []

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    kpts = result.keypoints.data.detach().cpu().numpy()
    candidates = []
    max_usable = 0
    max_strong = 0

    for box, conf, kp in zip(boxes, confs, kpts):
        if kp.ndim != 2 or kp.shape[1] < 3:
            continue
        kp_conf = kp[:, 2]
        strong = int((kp_conf >= 0.50).sum())
        usable = int((kp_conf >= 0.25).sum())
        max_usable = max(max_usable, usable)
        max_strong = max(max_strong, strong)
        conf = float(conf)

        # Tiered pose evidence. Very weak boxes still need a convincing skeleton,
        # but medium/high-confidence person boxes are not discarded just because
        # legs/arms are hidden by a desk, chair or another person.
        if conf < 0.08:
            if usable < 4 or strong < 2:
                continue
        elif conf < 0.20:
            if usable < 2 or strong < 1:
                continue
        elif usable < 1:
            continue

        x1, y1, x2, y2 = map(float, box)
        candidates.append(
            {
                "box": (x1, y1, x2, y2),
                "conf": conf,
                "quality": conf + 0.025 * usable + 0.015 * strong,
                "kp_xy": kp[:, :2].copy(),
                "valid_kp": (kp_conf >= 0.25).copy(),
            }
        )

    # Two real people may overlap. Suppress only near-identical pose duplicates.
    candidates.sort(key=lambda row: row["quality"], reverse=True)
    kept = []
    for cand in candidates:
        duplicate = False
        for old in kept:
            iou, containment = _overlap(cand["box"], old["box"])
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
                same_pose = float(np.median(distances) / scale) <= 0.10
            if same_pose and (iou >= 0.35 or containment >= 0.65):
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
        if len(kept) >= max_det:
            break

    raw_count = int(len(confs))
    raw_max = float(np.max(confs)) if raw_count else 0.0
    if (
        _DIAG_CALLS <= 3
        or _DIAG_CALLS % 10 == 0
        or (raw_count > 0 and not kept)
        or kept
    ):
        print(
            "CAMERA_POSE_DIAG "
            f"n={_DIAG_CALLS} raw={raw_count} kept={len(kept)} "
            f"raw_max={raw_max:.3f} max_usable={max_usable} max_strong={max_strong}",
            flush=True,
        )

    return [(row["box"], row["conf"]) for row in kept]


def yolo_pose_worker(job_q, result_q) -> None:
    """Spawn-safe CUDA YOLO26s-pose worker preserving CameraDetectionV2 IPC."""
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

        startup_delay = float(os.environ.get("CAMERA_V2_DETECT_STARTUP_DELAY", "0.5"))
        if startup_delay > 0:
            time.sleep(startup_delay)

        imgsz = int(os.environ.get("CAMERA_V2_POSE_IMGSZ", str(DEFAULT_IMGSZ)))
        conf = float(os.environ.get("CAMERA_V2_POSE_CONF", str(DEFAULT_CONF)))
        iou = float(os.environ.get("CAMERA_V2_POSE_IOU", str(DEFAULT_IOU)))
        max_det = int(det.MAX_DET)

        local_model = Path(__file__).resolve().parents[2] / "yolo26s-pose.pt"
        configured = os.environ.get("CAMERA_V2_POSE_MODEL", "").strip()
        if configured:
            model_spec = configured
        elif local_model.is_file():
            model_spec = str(local_model)
        else:
            model_spec = "yolo26s-pose.pt"

        model = YOLO(model_spec)
        warm = np.zeros((det.INFER_HEIGHT, det.INFER_WIDTH, 3), dtype=np.uint8)
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
                "backend": "YOLO26s-pose",
                "device": torch.cuda.get_device_name(0),
                "cuda": str(torch.version.cuda),
                "model": model_spec,
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
                        f"YOLO pose batch mismatch predictions={len(predictions)} cameras={len(job['cameras'])}"
                    )
                output = {
                    cid: _rows_from_result(prediction, max_det=max_det)
                    for cid, prediction in zip(job["cameras"], predictions)
                }
                result_q.put(
                    {
                        "type": "result",
                        "backend": "YOLO26s-pose",
                        "cameras": job["cameras"],
                        "captured": job["captured"],
                        "boxes": output,
                        "batch_ms": (ended - started) * 1000.0,
                    }
                )
            except torch.cuda.OutOfMemoryError as exc:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                result_q.put({"type": "batch_error", "error": f"YOLO26s-pose CUDA OOM: {exc}"})
            except BaseException as exc:
                result_q.put({"type": "batch_error", "error": f"YOLO26s-pose {type(exc).__name__}: {exc}"})
    except BaseException as exc:
        result_q.put({"type": "fatal", "error": f"YOLO26s-pose {type(exc).__name__}: {exc}"})


def _capture_gate_until_sample(self, _pad, _info, cid: str):
    with self.capture_lock:
        requested = bool(self.capture_requested.get(cid, False))
    return self.Gst.PadProbeReturn.OK if requested else self.Gst.PadProbeReturn.DROP


def install() -> None:
    """Install pose worker and strict JIT capture gate into CameraDetectionV2."""
    from . import detection

    detection._yolo_worker = yolo_pose_worker
    detection.CameraDetectionV2._infer_gate_probe = _capture_gate_until_sample
