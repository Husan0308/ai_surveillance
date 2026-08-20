from __future__ import annotations

import os
import time
from collections import defaultdict

import numpy as np


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _iou(a, b) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0.0 else 0.0


def _containment(a, b) -> float:
    inter = _intersection(a, b)
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller > 0.0 else 0.0


def _center_distance(a, b) -> float:
    acx = (a[0] + a[2]) * 0.5
    acy = (a[1] + a[3]) * 0.5
    bcx = (b[0] + b[2]) * 0.5
    bcy = (b[1] + b[3]) * 0.5
    scale = max(20.0, max(_area(a), _area(b)) ** 0.5)
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / scale


def _center_inside(box, bounds) -> bool:
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    return bounds[0] <= cx <= bounds[2] and bounds[1] <= cy <= bounds[3]


def _dedupe(boxes, *, iou_threshold: float, containment_threshold: float, center_threshold: float):
    ordered = sorted(boxes, key=lambda row: row[4], reverse=True)
    kept = []
    for candidate in ordered:
        if any(
            _iou(candidate, existing) >= iou_threshold
            or (
                _containment(candidate, existing) >= containment_threshold
                and _center_distance(candidate, existing) <= center_threshold
            )
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def _normalized_zone_to_pixels(zone, width: int, height: int):
    return (
        float(zone[0]) * width,
        float(zone[1]) * height,
        float(zone[2]) * width,
        float(zone[3]) * height,
    )


def _hard_reject(box, width: int, height: int, zones, max_box_height_ratio: float, overlap_threshold: float) -> bool:
    if not zones:
        return False
    box_h = max(0.0, box[3] - box[1]) / max(1.0, float(height))
    if box_h > max_box_height_ratio:
        return False
    box_area = max(1.0, _area(box))
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    for zone in zones:
        if len(zone) < 4:
            continue
        pixel_zone = _normalized_zone_to_pixels(zone, width, height)
        if pixel_zone[0] <= cx <= pixel_zone[2] and pixel_zone[1] <= cy <= pixel_zone[3]:
            return True
        if _intersection(box, pixel_zone) / box_area >= overlap_threshold:
            return True
    return False


def _filter_hard_masks(camera_id: str, boxes, width: int, height: int, hard_cfg: dict):
    zones = dict(hard_cfg.get("cameras") or {}).get(camera_id, [])
    if not zones:
        return list(boxes), 0
    max_h = float(hard_cfg.get("max_box_height_ratio", 0.30))
    overlap = float(hard_cfg.get("overlap_threshold", 0.15))
    kept = []
    rejected = 0
    for box in boxes:
        if _hard_reject(box, width, height, zones, max_h, overlap):
            rejected += 1
        else:
            kept.append(box)
    return kept, rejected


def _rotation_image(image: np.ndarray, rotation: int) -> np.ndarray:
    rotation = int(rotation) % 360
    if rotation == 0:
        return image
    if rotation == 90:
        return np.ascontiguousarray(np.rot90(image, k=3))
    if rotation == 180:
        return np.ascontiguousarray(np.rot90(image, k=2))
    if rotation == 270:
        return np.ascontiguousarray(np.rot90(image, k=1))
    raise ValueError(f"unsupported ROI rotation: {rotation}")


def _inverse_rotate_box(box, rotation: int, original_w: int, original_h: int):
    x1, y1, x2, y2, conf = box
    rotation = int(rotation) % 360
    if rotation == 0:
        return box
    if rotation == 90:
        ox1, oy1, ox2, oy2 = y1, original_h - x2, y2, original_h - x1
    elif rotation == 270:
        ox1, oy1, ox2, oy2 = original_w - y2, x1, original_w - y1, x2
    elif rotation == 180:
        ox1, oy1, ox2, oy2 = original_w - x2, original_h - y2, original_w - x1, original_h - y1
    else:
        raise ValueError(f"unsupported ROI rotation: {rotation}")
    return (
        max(0.0, min(float(original_w), ox1)),
        max(0.0, min(float(original_h), oy1)),
        max(0.0, min(float(original_w), ox2)),
        max(0.0, min(float(original_h), oy2)),
        conf,
    )


def _extract_person_rows(detections, person_class_id: int, confidence_floor: float, max_det: int):
    xyxy = np.asarray(getattr(detections, "xyxy", []), dtype=np.float32)
    class_ids = np.asarray(getattr(detections, "class_id", []))
    confidences = np.asarray(getattr(detections, "confidence", []), dtype=np.float32)
    rows = []
    for box, class_id, confidence in zip(xyxy, class_ids, confidences):
        confidence = float(confidence)
        if int(class_id) != person_class_id or confidence < confidence_floor:
            continue
        rows.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), confidence))
    rows.sort(key=lambda row: row[4], reverse=True)
    return rows[:max_det]


def _predict(model, images, threshold: float):
    predictions = model.predict(images, threshold=threshold)
    if isinstance(predictions, list):
        return predictions
    return [predictions]


def _roi_bounds(camera_cfg: dict, width: int, height: int):
    values = camera_cfg.get("box") or camera_cfg.get("roi")
    if not isinstance(values, (list, tuple)) or len(values) < 4:
        return None
    nx1, ny1, nx2, ny2 = [max(0.0, min(1.0, float(v))) for v in values[:4]]
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    x1 = max(0, min(width - 1, int(round(nx1 * width))))
    y1 = max(0, min(height - 1, int(round(ny1 * height))))
    x2 = max(x1 + 1, min(width, int(round(nx2 * width))))
    y2 = max(y1 + 1, min(height, int(round(ny2 * height))))
    return x1, y1, x2, y2


def rfdetr_worker_v2(job_q, result_q, cfg: dict) -> None:
    """RF-DETR-S worker using the mature Core-v1 detection policy.

    The model is different, but the proven production behavior is preserved:
    high-recall full-frame pass, selective high-resolution ROI recovery, hard
    false-positive masks, confidence-first full/ROI fusion, latest-only one-
    inflight scheduling (owned by the parent), and no detector failure may own
    the six-camera display path.
    """
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
        device = str(cfg.get("device", "cuda:0"))
        if device.startswith("cuda:"):
            torch.cuda.set_device(int(device.split(":", 1)[1]))
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

        model = RFDETRSmall(device=device)
        if bool(cfg.get("optimize_for_inference", False)):
            model.optimize_for_inference()

        threshold = float(cfg.get("threshold", 0.06))
        person_class_id = int(cfg.get("person_class_id", 1))
        max_det = max(1, int(cfg.get("max_det", 50)))
        camera_thresholds = {str(k): float(v) for k, v in dict(cfg.get("camera_thresholds") or {}).items()}

        duplicate_iou = float(cfg.get("duplicate_iou", 0.58))
        fusion_containment = float(cfg.get("fusion_containment", 0.84))
        fusion_center_distance = float(cfg.get("fusion_center_distance", 0.40))
        hard_cfg = dict(cfg.get("hard_exclusion") or {})
        roi_cfg = dict(cfg.get("roi_second_pass") or {})
        roi_enabled = bool(roi_cfg.get("enabled", False))
        roi_cameras = dict(roi_cfg.get("cameras") or {})
        roi_every_n = max(1, int(roi_cfg.get("every_n", 2)))
        global_trigger_max = max(0, int(roi_cfg.get("trigger_max_full_roi_persons", 1)))
        roi_seen = defaultdict(int)

        warm_h = int(cfg.get("capture_height", 432))
        warm_w = int(cfg.get("capture_width", 768))
        warm = np.zeros((warm_h, warm_w, 3), dtype=np.uint8)
        with torch.inference_mode():
            _predict(model, warm, threshold)

        result_q.put(
            {
                "type": "ready",
                "device": torch.cuda.get_device_name(torch.cuda.current_device()),
                "cuda": str(torch.version.cuda),
                "model": "RF-DETR-Small",
                "policy": "core-v1-full+roi+fusion",
            }
        )

        while True:
            job = job_q.get()
            if job is None:
                return
            started = time.monotonic()
            try:
                cameras = list(job["cameras"])
                frames_bgr = list(job["frames"])
                frames_rgb = [np.ascontiguousarray(frame[:, :, ::-1]) for frame in frames_bgr]

                with torch.inference_mode():
                    full_predictions = _predict(model, frames_rgb, threshold)

                boxes_by_camera = {}
                hard_rejects = 0
                frame_shapes = {}
                for cid, frame, prediction in zip(cameras, frames_rgb, full_predictions):
                    height, width = frame.shape[:2]
                    frame_shapes[cid] = (width, height)
                    floor = max(threshold, camera_thresholds.get(cid, threshold))
                    boxes = _extract_person_rows(prediction, person_class_id, floor, max_det)
                    boxes, rejected = _filter_hard_masks(cid, boxes, width, height, hard_cfg)
                    boxes_by_camera[cid] = boxes
                    hard_rejects += rejected

                roi_jobs = []
                if roi_enabled:
                    for cid, frame in zip(cameras, frames_rgb):
                        camera_cfg = dict(roi_cameras.get(cid) or {})
                        if not camera_cfg:
                            continue
                        roi_seen[cid] += 1
                        every_n = max(1, int(camera_cfg.get("every_n", roi_every_n)))
                        if roi_seen[cid] % every_n:
                            continue
                        height, width = frame.shape[:2]
                        bounds = _roi_bounds(camera_cfg, width, height)
                        if bounds is None:
                            continue
                        x1, y1, x2, y2 = bounds
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        full_inside = sum(1 for box in boxes_by_camera.get(cid, ()) if _center_inside(box, bounds))
                        mode = str(camera_cfg.get("mode", "augment")).lower()
                        trigger_max = int(camera_cfg.get("trigger_max_full_roi_persons", global_trigger_max))
                        should_run = bool(camera_cfg.get("always_run", False)) or mode == "verify" or full_inside <= trigger_max
                        if not should_run:
                            continue
                        rotations = camera_cfg.get("rotations", [0])
                        if not isinstance(rotations, (list, tuple)):
                            rotations = [rotations]
                        for rotation in rotations:
                            rotation = int(rotation) % 360
                            if rotation not in {0, 90, 180, 270}:
                                continue
                            roi_jobs.append(
                                {
                                    "camera_id": cid,
                                    "bounds": bounds,
                                    "mode": mode,
                                    "rotation": rotation,
                                    "accept_conf": float(camera_cfg.get("accept_conf", roi_cfg.get("accept_conf", threshold))),
                                    "image": _rotation_image(crop, rotation),
                                    "original_w": int(crop.shape[1]),
                                    "original_h": int(crop.shape[0]),
                                }
                            )

                roi_wall_ms = 0.0
                if roi_jobs:
                    roi_started = time.monotonic()
                    roi_images = [job_row["image"] for job_row in roi_jobs]
                    roi_threshold = min(threshold, min(float(row["accept_conf"]) for row in roi_jobs))
                    with torch.inference_mode():
                        roi_predictions = _predict(model, roi_images, roi_threshold)
                    grouped = defaultdict(list)
                    modes = {}
                    bounds_by_camera = {}
                    for row, prediction in zip(roi_jobs, roi_predictions):
                        cid = row["camera_id"]
                        modes[cid] = row["mode"]
                        bounds_by_camera[cid] = row["bounds"]
                        raw_boxes = _extract_person_rows(
                            prediction,
                            person_class_id,
                            float(row["accept_conf"]),
                            max_det,
                        )
                        xoff, yoff, _x2, _y2 = row["bounds"]
                        mapped = []
                        for raw_box in raw_boxes:
                            local = _inverse_rotate_box(
                                raw_box,
                                row["rotation"],
                                row["original_w"],
                                row["original_h"],
                            )
                            mapped.append(
                                (
                                    float(xoff) + local[0],
                                    float(yoff) + local[1],
                                    float(xoff) + local[2],
                                    float(yoff) + local[3],
                                    local[4],
                                )
                            )
                        width, height = frame_shapes[cid]
                        mapped, rejected = _filter_hard_masks(cid, mapped, width, height, hard_cfg)
                        hard_rejects += rejected
                        grouped[cid].extend(mapped)

                    for cid in set(row["camera_id"] for row in roi_jobs):
                        roi_boxes = grouped.get(cid, [])
                        if modes.get(cid) == "verify":
                            bounds = bounds_by_camera[cid]
                            outside = [box for box in boxes_by_camera.get(cid, []) if not _center_inside(box, bounds)]
                            boxes_by_camera[cid] = outside + roi_boxes
                        else:
                            boxes_by_camera.setdefault(cid, []).extend(roi_boxes)
                    roi_wall_ms = (time.monotonic() - roi_started) * 1000.0

                output = {}
                for cid in cameras:
                    fused = _dedupe(
                        boxes_by_camera.get(cid, []),
                        iou_threshold=duplicate_iou,
                        containment_threshold=fusion_containment,
                        center_threshold=fusion_center_distance,
                    )
                    output[cid] = [([box[0], box[1], box[2], box[3]], box[4]) for box in fused[:max_det]]

                result_q.put(
                    {
                        "type": "result",
                        "cameras": cameras,
                        "captured": list(job["captured"]),
                        "boxes": output,
                        "batch_ms": (time.monotonic() - started) * 1000.0,
                        "roi_ms": roi_wall_ms,
                        "roi_inputs": len({row["camera_id"] for row in roi_jobs}),
                        "roi_variants": len(roi_jobs),
                        "hard_rejects": hard_rejects,
                    }
                )
            except torch.cuda.OutOfMemoryError as exc:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                result_q.put({"type": "batch_error", "error": f"CUDA OOM: {exc}"})
            except BaseException as exc:
                result_q.put({"type": "batch_error", "error": f"{type(exc).__name__}: {exc}"})
    except BaseException as exc:
        result_q.put({"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
