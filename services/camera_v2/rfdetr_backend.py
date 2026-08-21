from __future__ import annotations

"""RF-DETR-S backend using the proven Core-v1 detection policy.

The detector model remains RF-DETR-S.  Around it we restore the logic from the
old ``rebuild/core-v1-clean`` period that the live deployment found reliable:

* aspect-matched full-frame inference;
* strict person-only output;
* latest-only/stale-result discipline;
* cross-pass deduplication;
* CAM-05 verification ROI;
* CAM-06 augmentation ROI and static false-positive exclusion;
* old adaptive Kalman/Byte presentation tracker;
* no optical flow, ReID or NvDCF in the Pascal path.

Presentation rectangles are deliberately NOT injected here before the display
tiler.  ``CameraPascalSafeRuntime`` maps source-space boxes to the final grid or
fullscreen wall and injects them immediately before nvdsosd.  That mirrors the
old-good direct-overlay semantics and avoids a second metadata coordinate
transform when the tiler layout changes.
"""

import importlib.metadata
import math
import os
import time

import numpy as np

from .old_good_rfdetr_tracker import OldGoodRFDETRBoxManager
from .person_candidate_filter import PersonCandidateFilter


# Core-v1 camera-specific detector policy.  Coordinates are normalized to the
# analysis/source image before CameraDetectionV2 scales them to mux coordinates.
_ROI_POLICY = {
    "CAM-05": {
        "mode": "verify",
        "box": (0.27, 0.00, 0.72, 0.54),
        "every_n": 2,
        "trigger_max": 1,
        "accept_conf": 0.11,
    },
    "CAM-06": {
        "mode": "augment",
        "box": (0.36, 0.12, 0.74, 0.48),
        "every_n": 2,
        "trigger_max": 0,
        "accept_conf": 0.075,
    },
}

_CAM06_EXCLUSION = (0.50, 0.00, 0.78, 0.22)


def _area(box) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def _intersection(a, b) -> float:
    return max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))) * max(
        0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))
    )


def _iou(a, b) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0.0 else 0.0


def _containment(a, b) -> float:
    inter = _intersection(a, b)
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller > 0.0 else 0.0


def _center_distance(a, b) -> float:
    acx = (float(a[0]) + float(a[2])) * 0.5
    acy = (float(a[1]) + float(a[3])) * 0.5
    bcx = (float(b[0]) + float(b[2])) * 0.5
    bcy = (float(b[1]) + float(b[3])) * 0.5
    scale = max(20.0, math.sqrt(max(_area(a), _area(b), 1.0)))
    return math.hypot(acx - bcx, acy - bcy) / scale


def _dedupe_rows(rows, max_det: int):
    """Core-v1 source-aware duplicate fusion after full + ROI inference."""
    ordered = sorted(rows, key=lambda item: float(item[1]), reverse=True)
    kept = []
    for box, score in ordered:
        duplicate = False
        for other, _other_score in kept:
            if _iou(box, other) >= 0.58 or (
                _containment(box, other) >= 0.84
                and _center_distance(box, other) <= 0.40
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append((list(map(float, box)), float(score)))
            if len(kept) >= max(1, int(max_det)):
                break
    return kept


def _center_inside(box, roi, width: int, height: int) -> bool:
    x1, y1, x2, y2 = roi
    cx = (float(box[0]) + float(box[2])) * 0.5 / max(1.0, float(width))
    cy = (float(box[1]) + float(box[3])) * 0.5 / max(1.0, float(height))
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _cam06_excluded(box, width: int, height: int) -> bool:
    """Reject the old known TV/static false-positive zone only for small boxes."""
    if height <= 0 or width <= 0:
        return False
    bx1, by1, bx2, by2 = [float(v) for v in box]
    box_h_ratio = max(0.0, by2 - by1) / float(height)
    if box_h_ratio > 0.30:
        return False
    zx1, zy1, zx2, zy2 = _CAM06_EXCLUSION
    zone = (zx1 * width, zy1 * height, zx2 * width, zy2 * height)
    area = _area(box)
    if area <= 0.0:
        return True
    return _intersection(box, zone) / area >= 0.15


def _filter_static_zone(cid: str, rows, width: int, height: int):
    if cid != "CAM-06":
        return rows
    return [(box, score) for box, score in rows if not _cam06_excluded(box, width, height)]


def _single_prediction(value):
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return value[0]
    return value


def rfdetr_worker(job_q, result_q) -> None:
    """Spawn-safe RF-DETR-S worker with old-good full/ROI person policy."""
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

        from . import detection as det

        startup_delay = float(os.environ.get("CAMERA_V2_DETECT_STARTUP_DELAY", "2.0"))
        if startup_delay > 0:
            time.sleep(startup_delay)

        capture_shape = (int(det.INFER_HEIGHT), int(det.INFER_WIDTH))
        # RF-DETR >=1.6.2 officially supports a non-square predict(shape=(h,w)).
        # Match the model resize to the 16:9 analysis tile instead of distorting
        # CCTV people to the default square operating point.
        model_shape = (
            int(os.environ.get("CAMERA_V2_RFDETR_MODEL_HEIGHT", str(capture_shape[0]))),
            int(os.environ.get("CAMERA_V2_RFDETR_MODEL_WIDTH", str(capture_shape[1]))),
        )
        roi_shape = (
            int(os.environ.get("CAMERA_V2_RFDETR_ROI_HEIGHT", "512")),
            int(os.environ.get("CAMERA_V2_RFDETR_ROI_WIDTH", "640")),
        )
        threshold = float(det.CONF)
        roi_threshold = float(os.environ.get("CAMERA_V2_RFDETR_ROI_CONF", "0.06"))
        max_det = int(det.MAX_DET)
        person_filter = PersonCandidateFilter()
        per_camera_runs: dict[str, int] = {}
        telemetry_budget = max(
            0, int(os.environ.get("CAMERA_V2_RFDETR_TRUTH_LOG_BUDGET", "120"))
        )

        model = RFDETRSmall(device="cuda:0")
        warm = np.zeros((capture_shape[0], capture_shape[1], 3), dtype=np.uint8)
        with torch.inference_mode():
            model.predict(
                warm,
                threshold=threshold,
                shape=model_shape,
                include_source_image=False,
            )

        try:
            version = importlib.metadata.version("rfdetr")
        except Exception:
            version = "unknown"

        print(
            "RFDETR_OLDGOOD_READY "
            f"version={version} model=RF-DETR-S device={torch.cuda.get_device_name(0)} "
            f"capture={capture_shape[1]}x{capture_shape[0]} "
            f"model_shape={model_shape[1]}x{model_shape[0]} "
            f"roi_shape={roi_shape[1]}x{roi_shape[0]} threshold={threshold:.2f} "
            f"roi_threshold={roi_threshold:.3f} person_ids={person_filter.person_ids} "
            "policy=core-v1-full+roi+dedupe+kalman-byte",
            flush=True,
        )
        result_q.put(
            {
                "type": "ready",
                "backend": "RF-DETR-S-old-good",
                "device": torch.cuda.get_device_name(0),
                "cuda": str(torch.version.cuda),
                "model": "RFDETRSmall",
                "version": version,
                "capture_shape": capture_shape,
                "model_shape": model_shape,
                "threshold": threshold,
            }
        )

        while True:
            job = job_q.get()
            if job is None:
                return
            started = time.monotonic()
            try:
                rgb_frames = [np.ascontiguousarray(frame[..., ::-1]) for frame in job["frames"]]
                with torch.inference_mode():
                    predictions = model.predict(
                        rgb_frames,
                        threshold=threshold,
                        shape=model_shape,
                        include_source_image=False,
                    )
                if not isinstance(predictions, (list, tuple)):
                    predictions = [predictions]
                if len(predictions) != len(job["cameras"]):
                    raise RuntimeError(
                        f"RF-DETR batch mismatch: predictions={len(predictions)} cameras={len(job['cameras'])}"
                    )

                output = {}
                telemetry = []
                for cid, rgb, prediction in zip(job["cameras"], rgb_frames, predictions):
                    full_rows, full_stats = person_filter.filter(prediction, max_det)
                    h, w = rgb.shape[:2]
                    full_rows = _filter_static_zone(cid, full_rows, w, h)
                    merged = list(full_rows)
                    roi_added = 0
                    roi_ran = False

                    policy = _ROI_POLICY.get(cid)
                    if policy is not None:
                        run_index = per_camera_runs.get(cid, 0) + 1
                        per_camera_runs[cid] = run_index
                        roi = policy["box"]
                        inside_count = sum(
                            1 for box, _score in full_rows if _center_inside(box, roi, w, h)
                        )
                        due = run_index % max(1, int(policy["every_n"])) == 0
                        trigger = inside_count <= int(policy["trigger_max"])
                        if due and trigger:
                            rx1 = max(0, min(w - 2, int(round(float(roi[0]) * w))))
                            ry1 = max(0, min(h - 2, int(round(float(roi[1]) * h))))
                            rx2 = max(rx1 + 2, min(w, int(round(float(roi[2]) * w))))
                            ry2 = max(ry1 + 2, min(h, int(round(float(roi[3]) * h))))
                            crop = np.ascontiguousarray(rgb[ry1:ry2, rx1:rx2])
                            if crop.size:
                                roi_ran = True
                                with torch.inference_mode():
                                    roi_pred = _single_prediction(
                                        model.predict(
                                            crop,
                                            threshold=roi_threshold,
                                            shape=roi_shape,
                                            include_source_image=False,
                                        )
                                    )
                                if roi_pred is not None:
                                    roi_rows, _roi_stats = person_filter.filter(roi_pred, max_det)
                                    mapped = []
                                    for box, score in roi_rows:
                                        if float(score) < float(policy["accept_conf"]):
                                            continue
                                        x1, y1, x2, y2 = [float(v) for v in box]
                                        mapped_box = [x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1]
                                        if cid == "CAM-06" and _cam06_excluded(mapped_box, w, h):
                                            continue
                                        mapped.append((mapped_box, float(score)))
                                    roi_added = len(mapped)
                                    if policy["mode"] == "verify":
                                        outside = [
                                            row for row in full_rows
                                            if not _center_inside(row[0], roi, w, h)
                                        ]
                                        merged = outside + mapped
                                    else:
                                        merged = list(full_rows) + mapped

                    merged = _dedupe_rows(merged, max_det)
                    output[cid] = merged
                    best = max((float(score) for _box, score in merged), default=0.0)
                    telemetry.append(
                        f"{cid}:{len(merged)}@{best:.2f}/raw{full_stats.raw}/roi{roi_added}{'*' if roi_ran else ''}"
                    )

                ended = time.monotonic()
                if telemetry_budget > 0:
                    telemetry_budget -= 1
                    print(
                        "RFDETR_OLDGOOD_RESULT "
                        f"batch={(ended-started)*1000.0:.1f}ms persons=[{' '.join(telemetry)}]",
                        flush=True,
                    )
                result_q.put(
                    {
                        "type": "result",
                        "backend": "RF-DETR-S-old-good",
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
                result_q.put({"type": "batch_error", "error": f"RF-DETR-S CUDA OOM: {exc}"})
            except BaseException as exc:
                result_q.put(
                    {"type": "batch_error", "error": f"RF-DETR-S {type(exc).__name__}: {exc}"}
                )
    except BaseException as exc:
        result_q.put({"type": "fatal", "error": f"RF-DETR-S {type(exc).__name__}: {exc}"})


def _capture_gate_until_sample(self, _pad, _info, cid: str):
    with self.capture_lock:
        requested = bool(self.capture_requested.get(cid, False))
    return self.Gst.PadProbeReturn.OK if requested else self.Gst.PadProbeReturn.DROP


def _no_pretiler_detection_meta(self, _pad, _info):
    """Old-good semantics: final presentation overlay is injected after tiling."""
    return self.Gst.PadProbeReturn.OK


def install() -> None:
    """Install RF-DETR-S + old-good Core-v1 policy into CameraDetectionV2."""
    selected = os.environ.get("CAMERA_V2_DETECT_BACKEND", "rfdetr-s").strip().lower()
    if selected in {"stable-yolo26m", "yolo26m", "yolo", "stable-yolo"}:
        from .stable_yolo_backend import install as install_stable_yolo
        install_stable_yolo()
        return
    if selected not in {"rfdetr-s", "rfdetr", "rf-detr-s", ""}:
        raise RuntimeError(f"unsupported CAMERA_V2_DETECT_BACKEND={selected!r}")

    from . import detection

    detection._yolo_worker = rfdetr_worker
    detection.SmoothBoxManager = OldGoodRFDETRBoxManager
    detection.CameraDetectionV2._infer_gate_probe = _capture_gate_until_sample
    detection.CameraDetectionV2._inject_boxes_probe = _no_pretiler_detection_meta

    print(
        "CAMERA_DETECT_BACKEND selected=rfdetr-s-old-good model=RF-DETR-S "
        "logic=core-v1-clean full+roi+dedupe+stale-gate+kalman-byte "
        "overlay=post-tiler-wall-space flow=OFF reid=OFF nvtracker=OFF",
        flush=True,
    )
