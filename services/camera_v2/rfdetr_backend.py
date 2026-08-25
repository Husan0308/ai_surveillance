from __future__ import annotations

"""RF-DETR-S detector backend for Camera V2.

The detector worker is process-isolated and preserves the existing CameraDetectionV2
job/result schema. Tracker selection is intentionally outside this module: the GTX
1050 Ti production controller constructs CameraPascalSafeRuntime directly, while
other runtimes may opt into the sparse NvDCF contract.
"""

import importlib.metadata
import os
import time

import numpy as np


def _person_mask(detections, class_id: np.ndarray) -> np.ndarray:
    """Resolve the person class across RF-DETR checkpoint label layouts."""

    data = getattr(detections, "data", None)
    if isinstance(data, dict):
        names = data.get("class_name")
        if names is not None:
            names = np.asarray(names).astype(str)
            if len(names) == len(class_id):
                normalized = np.char.lower(np.char.strip(names))
                return normalized == "person"

    # Pretrained COCO checkpoints may expose sparse category id 1 for person;
    # older/remapped or one-class checkpoints may expose zero-based id 0.
    return np.isin(class_id, (0, 1))


def _person_rows(detections, max_det: int) -> list[tuple[list[float], float]]:
    """Return person detections only, highest-confidence first."""

    xyxy = np.asarray(getattr(detections, "xyxy", []), dtype=np.float32)
    confidence = np.asarray(getattr(detections, "confidence", []), dtype=np.float32)
    class_id = np.asarray(getattr(detections, "class_id", []), dtype=np.int64)

    if xyxy.ndim != 2 or xyxy.shape[-1:] != (4,):
        return []
    if len(xyxy) != len(confidence) or len(xyxy) != len(class_id):
        return []

    indices = np.flatnonzero(_person_mask(detections, class_id))
    if not len(indices):
        return []
    indices = indices[np.argsort(confidence[indices])[::-1]]
    indices = indices[: max(1, int(max_det))]

    rows: list[tuple[list[float], float]] = []
    for idx in indices:
        box = xyxy[int(idx)]
        score = float(confidence[int(idx)])
        rows.append(([float(v) for v in box], score))
    return rows



def _dedupe_person_rows(
    rows,
    iou_gate: float = 0.62,
    containment_gate: float = 0.90,
    center_gate: float = 0.35,
):
    """Suppress overlapping duplicate person detections before tracking."""
    import math

    def area(box):
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    def intersection(a, b):
        return (
            max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        )

    def iou(a, b):
        inter = intersection(a, b)
        union = area(a) + area(b) - inter
        return inter / union if union > 0.0 else 0.0

    def containment(a, b):
        inter = intersection(a, b)
        smaller = min(area(a), area(b))
        return inter / smaller if smaller > 0.0 else 0.0

    def center_distance(a, b):
        acx = (a[0] + a[2]) * 0.5
        acy = (a[1] + a[3]) * 0.5
        bcx = (b[0] + b[2]) * 0.5
        bcy = (b[1] + b[3]) * 0.5

        aw = max(1.0, a[2] - a[0])
        ah = max(1.0, a[3] - a[1])
        bw = max(1.0, b[2] - b[0])
        bh = max(1.0, b[3] - b[1])

        scale = max(
            20.0,
            math.hypot(aw, ah),
            math.hypot(bw, bh),
        )
        return math.hypot(acx - bcx, acy - bcy) / scale

    ordered = sorted(rows, key=lambda row: float(row[1]), reverse=True)
    kept = []
    rejected = 0

    for box, conf in ordered:
        duplicate = False

        for kept_box, _kept_conf in kept:
            pair_iou = iou(box, kept_box)
            pair_containment = containment(box, kept_box)
            pair_center = center_distance(box, kept_box)

            if (
                pair_iou >= iou_gate
                or (
                    pair_containment >= containment_gate
                    and pair_center <= center_gate
                )
            ):
                duplicate = True
                break

        if duplicate:
            rejected += 1
        else:
            kept.append((box, conf))

    return kept, rejected


def _filter_bottom_fragments(
    rows,
    frame_w: float,
    frame_h: float,
):
    """Reject RF-DETR bottom-edge body fragments, not normal seated people."""
    kept = []
    rejected = 0

    for box, conf in rows:
        x1, y1, x2, y2 = [float(v) for v in box]
        width = max(1e-6, x2 - x1)
        height = max(1e-6, y2 - y1)

        bottom_ratio = y2 / max(1.0, frame_h)
        height_ratio = height / max(1.0, frame_h)
        aspect = width / height

        is_fragment = (
            bottom_ratio >= 0.985
            and height_ratio <= 0.12
            and aspect >= 1.60
        )

        if is_fragment:
            rejected += 1
            continue

        kept.append((box, conf))

    return kept, rejected


def rfdetr_worker(job_q, result_q) -> None:
    """Spawn-safe CUDA RF-DETR-S worker."""

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
        from rfdetr import RFDETRSmall

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        torch.cuda.set_device(0)
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

        # Import after the child owns CUDA. This keeps the parent Qt/DeepStream
        # process free of detector-side CUDA model state.
        from . import detection as det

        startup_delay = float(os.environ.get("CAMERA_V2_DETECT_STARTUP_DELAY", "3.0"))
        if startup_delay > 0:
            time.sleep(startup_delay)

        infer_shape = (int(det.INFER_HEIGHT), int(det.INFER_WIDTH))
        threshold = float(det.CONF)
        max_det = int(det.MAX_DET)

        model = RFDETRSmall(device="cuda:0")

        # Eager FP32 is the compatibility baseline for the Pascal deployment.
        warm = np.zeros((infer_shape[0], infer_shape[1], 3), dtype=np.uint8)
        with torch.inference_mode():
            model.predict(
                warm,
                threshold=threshold,
                shape=infer_shape,
                include_source_image=False,
            )

        try:
            version = importlib.metadata.version("rfdetr")
        except Exception:
            version = "unknown"

        result_q.put(
            {
                "type": "ready",
                "backend": "RF-DETR-S",
                "device": torch.cuda.get_device_name(0),
                "cuda": str(torch.version.cuda),
                "model": "RFDETRSmall",
                "version": version,
                "shape": infer_shape,
                "threshold": threshold,
            }
        )

        while True:
            job = job_q.get()
            if job is None:
                return

            started = time.monotonic()
            try:
                # The GStreamer side branch supplies BGR; RF-DETR NumPy input is RGB.
                rgb_frames = [
                    np.ascontiguousarray(frame[..., ::-1]) for frame in job["frames"]
                ]
                with torch.inference_mode():
                    predictions = model.predict(
                        rgb_frames,
                        threshold=threshold,
                        shape=infer_shape,
                        include_source_image=False,
                    )
                ended = time.monotonic()

                if not isinstance(predictions, (list, tuple)):
                    predictions = [predictions]
                if len(predictions) != len(job["cameras"]):
                    raise RuntimeError(
                        f"RF-DETR batch mismatch: predictions={len(predictions)} "
                        f"cameras={len(job['cameras'])}"
                    )

                output = {}
                fragment_rejected = {}

                for cid, prediction, source_frame in zip(
                    job["cameras"],
                    predictions,
                    job["frames"],
                ):
                    rows = _person_rows(prediction, max_det)

                    rows, duplicate_rejected = _dedupe_person_rows(
                        rows,
                        0.62,
                        0.90,
                        0.35,
                    )

                    source_h, source_w = source_frame.shape[:2]

                    rows, rejected = _filter_bottom_fragments(
                        rows,
                        float(source_w),
                        float(source_h),
                    )

                    output[cid] = rows
                    fragment_rejected[cid] = rejected
                result_q.put(
                    {
                        "type": "result",
                        "backend": "RF-DETR-S",
                        "cameras": job["cameras"],
                        "captured": job["captured"],
                        "boxes": output,
                        "fragment_rejected": fragment_rejected,
                        "batch_ms": (ended - started) * 1000.0,
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
                        "error": f"RF-DETR-S CUDA OOM: {exc}",
                    }
                )
            except BaseException as exc:
                result_q.put(
                    {
                        "type": "batch_error",
                        "error": f"RF-DETR-S {type(exc).__name__}: {exc}",
                    }
                )
    except BaseException as exc:
        result_q.put(
            {
                "type": "fatal",
                "error": f"RF-DETR-S {type(exc).__name__}: {exc}",
            }
        )


def _capture_gate_until_sample(self, _pad, _info, cid: str):
    """Pass buffers while a capture request is armed; scheduler clears the gate."""

    with self.capture_lock:
        requested = bool(self.capture_requested.get(cid, False))
    if not requested:
        return self.Gst.PadProbeReturn.DROP
    return self.Gst.PadProbeReturn.OK


def install() -> None:
    """Install RF-DETR-S into CameraDetectionV2 without selecting a tracker."""

    from . import detection

    detection._yolo_worker = rfdetr_worker
    detection.CameraDetectionV2._infer_gate_probe = _capture_gate_until_sample

    # Supported/non-Pascal runtimes may still use the existing sparse external
    # detector contract with NvDCF. The production GTX 1050 Ti controller never
    # imports those tracker runtimes, so safe mode has no tracker side effect.
    pascal_safe = os.environ.get("CAMERA_V2_PASCAL_SAFE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not pascal_safe:
        from .sparse_tracker_contract import install_sparse_tracker_contract

        install_sparse_tracker_contract()
