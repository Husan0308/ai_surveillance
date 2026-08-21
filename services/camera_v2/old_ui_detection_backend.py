from __future__ import annotations

"""Detection backend restored from the old Apsidal UI snapshot.

Canonical source: branch ``ui-aspect-ratio-final`` (commit 865bfedf...).
The model and policy intentionally match that UI-era Core-v1 detector:
YOLO26m, 704x448, conf=0.06, IoU=0.50, max_det=50, micro-batch=2,
latest-only freshness, CAM-05 verify ROI, CAM-06 augment ROI/static exclusion,
Core-v1 duplicate fusion, and the exact old visual Kalman/Byte tracker.
"""

import math
import os
import queue as pyqueue
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .old_ui_visual_tracker import VisualBox, VisualTracker


FULL_CONF = 0.06
FULL_IOU = 0.50
FULL_MAX_DET = 50
MODEL_WIDTH = 704
MODEL_HEIGHT = 448
ROI_WIDTH = 640
ROI_HEIGHT = 512
ROI_CONF = 0.045
ROI_IOU = 0.50
ROI_MAX_DET = 20
MAX_SUBMIT_AGE_SEC = 0.300
MAX_RESULT_AGE_SEC = 0.900

ROI_POLICY = {
    "CAM-05": {
        "mode": "verify",
        "box": (0.27, 0.00, 0.72, 0.54),
        "every_n": 2,
        "always_run": False,
        "accept_conf": 0.11,
        "trigger_max_full_roi_persons": 1,
        "rotations": (0,),
    },
    "CAM-06": {
        "mode": "augment",
        "box": (0.36, 0.12, 0.74, 0.48),
        "every_n": 2,
        "always_run": False,
        "accept_conf": 0.075,
        "trigger_max_full_roi_persons": 0,
        "rotations": (0,),
    },
}
HARD_EXCLUSION = {
    "CAM-06": ((0.50, 0.00, 0.78, 0.22),),
}


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def _iou(a, b):
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0.0 else 0.0


def _containment(a, b):
    inter = _intersection(a, b)
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller > 0.0 else 0.0


def _center_distance(a, b):
    acx = (a[0] + a[2]) * 0.5
    acy = (a[1] + a[3]) * 0.5
    bcx = (b[0] + b[2]) * 0.5
    bcy = (b[1] + b[3]) * 0.5
    scale = max(20.0, max(_area(a), _area(b)) ** 0.5)
    return math.hypot(acx - bcx, acy - bcy) / scale


def _deduplicate_boxes(boxes):
    ordered = sorted(boxes, key=lambda item: item[4], reverse=True)
    kept = []
    for candidate in ordered:
        if any(
            _iou(candidate, existing) >= 0.58
            or (
                _containment(candidate, existing) >= 0.84
                and _center_distance(candidate, existing) <= 0.40
            )
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def _box_center_inside(box, bounds):
    x1, y1, x2, y2 = bounds
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _normalized_zone_to_pixels(zone, source_w, source_h):
    return (
        float(zone[0]) * source_w,
        float(zone[1]) * source_h,
        float(zone[2]) * source_w,
        float(zone[3]) * source_h,
    )


def _hard_reject(box, source_w: int, source_h: int, zones):
    if not zones:
        return False
    box_h = max(0.0, box[3] - box[1]) / max(1.0, float(source_h))
    if box_h > 0.30:
        return False
    box_area = max(1.0, _area(box))
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    for zone in zones:
        z = _normalized_zone_to_pixels(zone, source_w, source_h)
        if z[0] <= cx <= z[2] and z[1] <= cy <= z[3]:
            return True
        if _intersection(box, z) / box_area >= 0.15:
            return True
    return False


def _filter_hard_masks(camera_id, boxes, source_w, source_h):
    zones = HARD_EXCLUSION.get(camera_id, ())
    if not zones:
        return boxes, 0
    kept = []
    rejected = 0
    for box in boxes:
        if _hard_reject(box, source_w, source_h, zones):
            rejected += 1
        else:
            kept.append(box)
    return kept, rejected


def _raw_prediction_boxes(prediction):
    pred_boxes = getattr(prediction, "boxes", None)
    if pred_boxes is None or not len(pred_boxes):
        return []
    xyxy = pred_boxes.xyxy.detach().cpu().tolist()
    confs = pred_boxes.conf.detach().cpu().tolist()
    return [
        (float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(conf))
        for c, conf in zip(xyxy, confs)
    ]


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


def _rotate_image(image, rotation: int):
    import cv2

    rotation = int(rotation) % 360
    if rotation == 0:
        return image
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError(f"unsupported ROI rotation: {rotation}")


def _map_roi_boxes(prediction, bounds, rotation: int):
    rx1, ry1, rx2, ry2 = bounds
    sx = max(1.0, float(rx2 - rx1)) / float(ROI_WIDTH)
    sy = max(1.0, float(ry2 - ry1)) / float(ROI_HEIGHT)
    mapped = []
    for raw in _raw_prediction_boxes(prediction):
        x1, y1, x2, y2, conf = _inverse_rotate_box(
            raw, rotation, ROI_WIDTH, ROI_HEIGHT
        )
        mapped.append(
            (
                float(rx1) + x1 * sx,
                float(ry1) + y1 * sy,
                float(rx1) + x2 * sx,
                float(ry1) + y2 * sy,
                conf,
            )
        )
    return mapped


def _resolve_model(spec: str) -> str:
    p = Path(spec).expanduser()
    if p.is_file():
        return str(p)
    root = Path(__file__).resolve().parents[2]
    local = root / p
    return str(local) if local.is_file() else str(spec)


def old_ui_yolo_worker(job_q, result_q) -> None:
    """YOLO process matching the old UI Core-v1 detector settings."""
    try:
        try:
            os.nice(8)
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

        import cv2
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(0)
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

        delay = float(os.environ.get("CAMERA_V2_DETECT_STARTUP_DELAY", "2.0"))
        if delay > 0:
            time.sleep(delay)

        model_path = _resolve_model(os.environ.get("CAMERA_V2_YOLO_MODEL", "models/yolo26m.pt"))
        model = YOLO(model_path)
        full_kwargs = {
            "imgsz": (MODEL_HEIGHT, MODEL_WIDTH),
            "conf": FULL_CONF,
            "iou": FULL_IOU,
            "classes": [0],
            "max_det": FULL_MAX_DET,
            "device": "cuda:0",
            "quantize": 32,
            "verbose": False,
        }
        roi_kwargs = dict(full_kwargs)
        roi_kwargs.update(
            {
                "imgsz": (ROI_HEIGHT, ROI_WIDTH),
                "conf": ROI_CONF,
                "iou": ROI_IOU,
                "max_det": ROI_MAX_DET,
            }
        )

        warm = np.zeros((MODEL_HEIGHT, MODEL_WIDTH, 3), dtype=np.uint8)
        with torch.inference_mode():
            model.predict(source=[warm, warm.copy()], **full_kwargs)

        roi_seen = {cid: 0 for cid in ROI_POLICY}
        print(
            "OLD_UI_YOLO_READY "
            f"model={model_path} device={torch.cuda.get_device_name(0)} "
            f"capture={MODEL_WIDTH}x{MODEL_HEIGHT} imgsz={MODEL_WIDTH}x{MODEL_HEIGHT} "
            f"conf={FULL_CONF:.2f} iou={FULL_IOU:.2f} max_det={FULL_MAX_DET} "
            "batch=2 roi=CAM05-verify+CAM06-augment tracker=core-v1-old-ui",
            flush=True,
        )
        result_q.put(
            {
                "type": "ready",
                "backend": "old-ui-yolo26m",
                "device": torch.cuda.get_device_name(0),
                "cuda": str(torch.version.cuda),
                "model": model_path,
            }
        )

        while True:
            job = job_q.get()
            if job is None:
                return
            started = time.monotonic()
            try:
                frames = [np.ascontiguousarray(frame) for frame in job["frames"]]
                with torch.inference_mode():
                    full_predictions = model.predict(source=frames, **full_kwargs)

                boxes_by_camera = {}
                hard_rejects = 0
                for cid, frame, prediction in zip(job["cameras"], frames, full_predictions):
                    h, w = frame.shape[:2]
                    boxes = _raw_prediction_boxes(prediction)
                    boxes, rejected = _filter_hard_masks(cid, boxes, w, h)
                    boxes_by_camera[cid] = boxes
                    hard_rejects += rejected

                roi_jobs = []
                for cid, frame in zip(job["cameras"], frames):
                    policy = ROI_POLICY.get(cid)
                    if policy is None:
                        continue
                    roi_seen[cid] = roi_seen.get(cid, 0) + 1
                    if roi_seen[cid] % max(1, int(policy["every_n"])):
                        continue
                    h, w = frame.shape[:2]
                    nx1, ny1, nx2, ny2 = policy["box"]
                    x1 = max(0, min(w - 1, int(round(nx1 * w))))
                    y1 = max(0, min(h - 1, int(round(ny1 * h))))
                    x2 = max(x1 + 1, min(w, int(round(nx2 * w))))
                    y2 = max(y1 + 1, min(h, int(round(ny2 * h))))
                    bounds = (x1, y1, x2, y2)
                    full_inside = sum(
                        1 for box in boxes_by_camera.get(cid, ())
                        if _box_center_inside(box, bounds)
                    )
                    should_run = (
                        bool(policy["always_run"])
                        or str(policy["mode"]).lower() == "verify"
                        or full_inside <= int(policy["trigger_max_full_roi_persons"])
                    )
                    if not should_run:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    resized = cv2.resize(crop, (ROI_WIDTH, ROI_HEIGHT), interpolation=cv2.INTER_LINEAR)
                    for rotation in policy["rotations"]:
                        roi_jobs.append(
                            (cid, bounds, policy, int(rotation), _rotate_image(resized, int(rotation)))
                        )

                roi_started = time.monotonic()
                if roi_jobs:
                    with torch.inference_mode():
                        roi_predictions = model.predict(
                            source=[item[4] for item in roi_jobs], **roi_kwargs
                        )
                    grouped = {}
                    for (cid, bounds, policy, rotation, _image), prediction in zip(
                        roi_jobs, roi_predictions
                    ):
                        mapped = [
                            box
                            for box in _map_roi_boxes(prediction, bounds, rotation)
                            if box[4] >= float(policy["accept_conf"])
                        ]
                        frame_index = job["cameras"].index(cid)
                        h, w = frames[frame_index].shape[:2]
                        mapped, rejected = _filter_hard_masks(cid, mapped, w, h)
                        hard_rejects += rejected
                        grouped.setdefault(cid, []).extend(mapped)

                    for cid, frame in zip(job["cameras"], frames):
                        policy = ROI_POLICY.get(cid)
                        if policy is None:
                            continue
                        h, w = frame.shape[:2]
                        nx1, ny1, nx2, ny2 = policy["box"]
                        bounds = (
                            int(round(nx1 * w)), int(round(ny1 * h)),
                            int(round(nx2 * w)), int(round(ny2 * h)),
                        )
                        roi_boxes = grouped.get(cid, [])
                        if str(policy["mode"]).lower() == "verify":
                            outside = [
                                box for box in boxes_by_camera.get(cid, [])
                                if not _box_center_inside(box, bounds)
                            ]
                            boxes_by_camera[cid] = outside + roi_boxes
                        else:
                            boxes_by_camera.setdefault(cid, []).extend(roi_boxes)
                roi_ms = (time.monotonic() - roi_started) * 1000.0 if roi_jobs else 0.0

                output = {}
                summary = []
                total = 0
                for cid in job["cameras"]:
                    boxes = _deduplicate_boxes(boxes_by_camera.get(cid, []))
                    rows = [([b[0], b[1], b[2], b[3]], b[4]) for b in boxes]
                    output[cid] = rows
                    total += len(rows)
                    best = max((float(row[1]) for row in rows), default=0.0)
                    summary.append(f"{cid}:{len(rows)}@{best:.2f}")

                batch_ms = (time.monotonic() - started) * 1000.0
                print(
                    "OLD_UI_YOLO_RESULT "
                    f"batch={batch_ms:.1f}ms roi={roi_ms:.1f}ms "
                    f"hard_rejects={hard_rejects} persons=[{' '.join(summary)}]",
                    flush=True,
                )
                result_q.put(
                    {
                        "type": "result",
                        "backend": "old-ui-yolo26m",
                        "cameras": list(job["cameras"]),
                        "captured": list(job["captured"]),
                        "boxes": output,
                        "batch_ms": batch_ms,
                        "detections": total,
                        "hard_rejects": hard_rejects,
                        "roi_inputs": len({item[0] for item in roi_jobs}),
                        "roi_variants": len(roi_jobs),
                    }
                )
            except BaseException as exc:
                result_q.put(
                    {"type": "batch_error", "error": f"old-ui YOLO26m {type(exc).__name__}: {exc}"}
                )
    except BaseException as exc:
        result_q.put({"type": "fatal", "error": f"old-ui YOLO26m {type(exc).__name__}: {exc}"})


class OldUIBoxManager:
    """Adapter around the exact ui-aspect-ratio-final VisualTracker blob."""

    _FRAGMENT = {"CAM-03", "CAM-05", "CAM-06"}
    _START_CONF = {"CAM-05": 0.30, "CAM-06": 0.38}
    _LOW_CONF = {"CAM-05": 0.18}
    _BIRTH_ZONES = {
        "CAM-05": ((0.27, 0.00, 0.72, 0.54, 0.38),),
        "CAM-06": ((0.36, 0.12, 0.76, 0.49, 0.42),),
    }
    _EXCLUSION = {"CAM-06": ((0.50, 0.00, 0.78, 0.22),)}

    def __init__(self, width: int, height: int) -> None:
        self.width = float(width)
        self.height = float(height)
        import threading
        self.lock = threading.RLock()
        self._trackers: dict[str, VisualTracker] = {}
        self._frame_ids: dict[str, int] = {}
        self.max_age = 0.800
        self.stale_results = 0

    def _new_tracker(self, cid: str) -> VisualTracker:
        return VisualTracker(
            hold_ms=800,
            memory_ms=3000,
            prediction_ms=420,
            match_iou=0.12,
            reacquire_distance=0.85,
            byte_match_center=0.70,
            byte_second_match_iou=0.04,
            byte_second_match_center=0.50,
            low_match_max_age_ms=650,
            byte_high_conf=0.24,
            byte_low_conf=0.08,
            low_conf_confirm=self._LOW_CONF.get(cid, 0.08),
            process_noise=0.85,
            measurement_noise=0.90,
            velocity_damping=0.96,
            size_velocity_damping=0.60,
            max_prediction_shift_boxes=0.55,
            max_prediction_size_ratio=0.08,
            adaptive_error_low=0.08,
            adaptive_error_high=0.25,
            center_response_slow=0.42,
            center_response_fast=0.88,
            size_response=0.30,
            snap_distance_boxes=0.65,
            reversal_damping=0.15,
            duplicate_iou=0.68,
            duplicate_containment=0.90,
            duplicate_center_distance=0.20,
            fragment_duplicate=cid in self._FRAGMENT,
            fragment_horizontal_overlap=0.78,
            fragment_x_center=0.18,
            fragment_max_area_ratio=0.55,
            fragment_min_vertical_overlap=0.20,
            fragment_max_vertical_gap=0.06,
            start_conf=self._START_CONF.get(cid, 0.34),
            new_track_min_conf=0.24,
            strong_confirm_hits=2,
            weak_confirm_hits=3,
            new_track_zones=self._BIRTH_ZONES.get(cid, ()),
            exclusion_zones=self._EXCLUSION.get(cid, ()),
            exclusion_max_box_height=0.30,
            exclusion_overlap_threshold=0.15,
        )

    def _tracker(self, cid: str) -> VisualTracker:
        tracker = self._trackers.get(cid)
        if tracker is None:
            tracker = self._new_tracker(cid)
            self._trackers[cid] = tracker
        return tracker

    def update(self, cid: str, captured_t: float, detections) -> None:
        now = time.monotonic()
        if now - float(captured_t) > MAX_RESULT_AGE_SEC:
            with self.lock:
                self.stale_results += 1
            return
        boxes = []
        for box, confidence in detections or ():
            try:
                x1, y1, x2, y2 = map(float, box)
                conf = float(confidence)
            except (TypeError, ValueError, OverflowError):
                continue
            boxes.append(VisualBox(x1, y1, x2, y2, conf))
        with self.lock:
            frame_id = self._frame_ids.get(cid, 0) + 1
            self._frame_ids[cid] = frame_id
            result = SimpleNamespace(
                frame_id=frame_id,
                frame_captured_monotonic=float(captured_t),
                boxes=tuple(boxes),
            )
            self._tracker(cid).update(
                result,
                now=now,
                source_width=self.width,
                source_height=self.height,
            )

    def render(self, cid: str, now: float):
        with self.lock:
            tracker = self._trackers.get(cid)
            if tracker is None:
                return []
            return [
                (float(b.x1), float(b.y1), float(b.x2), float(b.y2), float(b.confidence))
                for b in tracker.visible(float(now), target_time=float(now))
            ]

    @property
    def tracks(self):
        output = {}
        with self.lock:
            for cid, tracker in self._trackers.items():
                rows = {}
                with tracker._lock:
                    for tid, track in tracker._tracks.items():
                        rows[int(tid)] = SimpleNamespace(
                            last_det_t=float(track.last_observation),
                            confidence=float(track.confidence),
                        )
                output[cid] = rows
        return output


def _capture_gate_until_sample(self, _pad, _info, cid: str):
    with self.capture_lock:
        requested = bool(self.capture_requested.get(cid, False))
    return self.Gst.PadProbeReturn.OK if requested else self.Gst.PadProbeReturn.DROP


def _source_rows_to_wall(self, source_id: int, rows):
    if not rows:
        return []
    try:
        focus = int(self.tiler.get_property("show-source"))
    except Exception:
        focus = -1
    wall_w = float(max(1, int(self.wall_width)))
    wall_h = float(max(1, int(self.wall_height)))
    src_w = float(max(1, int(self.frame_width)))
    src_h = float(max(1, int(self.frame_height)))
    if focus >= 0:
        if int(source_id) != focus:
            return []
        left = top = 0.0
        tile_w, tile_h = wall_w, wall_h
    else:
        columns = max(1, int(getattr(self, "tiler_columns", 2)))
        rows_n = max(1, int(getattr(self, "tiler_rows", 3)))
        column = int(source_id) % columns
        row = int(source_id) // columns
        if row >= rows_n:
            return []
        tile_w = wall_w / columns
        tile_h = wall_h / rows_n
        left = column * tile_w
        top = row * tile_h
    sx = tile_w / src_w
    sy = tile_h / src_h
    output = []
    for x1, y1, x2, y2, confidence in rows:
        output.append(
            (
                left + float(x1) * sx,
                top + float(y1) * sy,
                left + float(x2) * sx,
                top + float(y2) * sy,
                float(confidence),
            )
        )
    return output


def _post_tiler_overlay_probe(self, _pad, info):
    buffer = info.get_buffer()
    if buffer is None:
        return self.Gst.PadProbeReturn.OK
    now = time.monotonic()
    added = 0
    requested = 0
    for cid, source_id in self.camera_index.items():
        wall_rows = _source_rows_to_wall(self, int(source_id), self.boxes.render(cid, now))
        if not wall_rows:
            continue
        requested += len(wall_rows)
        result = self.bridge.add_boxes(buffer, int(source_id), wall_rows)
        if result > 0:
            added += int(result)
    if added:
        with self.det_lock:
            self.meta_boxes += added
    budget = int(getattr(self, "_old_ui_overlay_log_budget", 20))
    if requested and budget > 0:
        self._old_ui_overlay_log_budget = budget - 1
        print(
            f"OLD_UI_OVERLAY requested={requested} injected={added} "
            f"wall={self.wall_width}x{self.wall_height} stage=post-tiler-pre-osd",
            flush=True,
        )
    return self.Gst.PadProbeReturn.OK


def _no_pre_tiler_meta(self, _pad, _info):
    return self.Gst.PadProbeReturn.OK


def _old_ui_scheduler(self) -> None:
    assert self.result_q is not None and self.job_q is not None
    try:
        ready = self.result_q.get(timeout=40.0)
    except pyqueue.Empty:
        with self.det_lock:
            self.det_error = "old UI YOLO worker startup timeout"
        return
    if ready.get("type") != "ready":
        with self.det_lock:
            self.det_error = ready.get("error", "old UI YOLO worker failed")
        return
    with self.det_lock:
        self.det_ready = True
        self.det_duty = 1.0
    print(
        "CAMERA_DETECT ready: old-ui YOLO26m micro_batch=2 input=704x448 "
        f"device={ready.get('device')} cuda={ready.get('cuda')} "
        "freshness=300/900ms tracker=old-ui-core-v1",
        flush=True,
    )

    ids = [camera.camera_id for camera in self.cameras]
    groups = [ids[i : i + 2] for i in range(0, len(ids), 2)]
    versions = {cid: 0 for cid in ids}
    group_index = 0
    while not self.det_stop.is_set():
        group = groups[group_index % len(groups)]
        group_index += 1
        self._request_group(group)
        rows = self.mailbox.wait_group(group, versions, timeout=1.5)
        if rows is None:
            self._clear_requests()
            with self.det_lock:
                self.capture_timeouts += 1
            self.det_stop.wait(0.05)
            continue

        now = time.monotonic()
        fresh_cameras = []
        fresh_frames = []
        fresh_captured = []
        for cid, row in zip(group, rows):
            version, captured_t, frame = row
            versions[cid] = version
            if now - float(captured_t) > MAX_SUBMIT_AGE_SEC:
                continue
            fresh_cameras.append(cid)
            fresh_frames.append(frame)
            fresh_captured.append(captured_t)
        self._clear_requests()
        if not fresh_cameras:
            self.det_stop.wait(0.005)
            continue

        try:
            self.job_q.put(
                {
                    "cameras": fresh_cameras,
                    "frames": fresh_frames,
                    "captured": fresh_captured,
                },
                timeout=0.5,
            )
            result = self.result_q.get(timeout=8.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "old UI YOLO result timeout"
            self.det_stop.wait(0.05)
            continue
        if result.get("type") == "fatal":
            with self.det_lock:
                self.det_error = result.get("error", "old UI YOLO fatal")
            return
        if result.get("type") == "batch_error":
            with self.det_lock:
                self.det_error = result.get("error", "old UI YOLO batch error")
            self.det_stop.wait(0.05)
            continue
        if result.get("type") != "result":
            continue

        counts = {}
        finish_now = time.monotonic()
        for cid, captured_t in zip(result["cameras"], result["captured"]):
            if finish_now - float(captured_t) > MAX_RESULT_AGE_SEC:
                counts[cid] = 0
                continue
            dets = self._scaled_detections(result["boxes"].get(cid, []))
            counts[cid] = len(dets)
            self.boxes.update(cid, captured_t, dets)

        with self.det_lock:
            self.det_calls += 1
            self.det_inputs += len(result["cameras"])
            self.det_batch_ms = float(result.get("batch_ms") or 0.0)
            self.det_counts.update(counts)
            self.det_error = ""
        self.det_stop.wait(0.005)


def _install_overlay(detection_module) -> None:
    cls = detection_module.CameraDetectionV2
    if getattr(cls, "_old_ui_overlay_installed", False):
        return
    cls._old_ui_post_tiler_overlay_probe = _post_tiler_overlay_probe
    original_init = cls.__init__

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        sink_pad = self.osd.get_static_pad("sink")
        if sink_pad is None:
            raise RuntimeError("old UI detection cannot access nvdsosd sink pad")
        self._old_ui_overlay_log_budget = 20
        sink_pad.add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._old_ui_post_tiler_overlay_probe,
        )
        print(
            "OLD_UI_OVERLAY_READY mapping=source-to-grid-or-fullscreen "
            "stage=post-tiler-pre-osd",
            flush=True,
        )

    cls.__init__ = wrapped_init
    cls._old_ui_overlay_installed = True


def install() -> None:
    from . import detection

    if (detection.INFER_WIDTH, detection.INFER_HEIGHT, detection.MICRO_BATCH) != (
        MODEL_WIDTH,
        MODEL_HEIGHT,
        2,
    ):
        raise RuntimeError(
            "old UI detection requires CAMERA_V2_DETECT_WIDTH=704 "
            "CAMERA_V2_DETECT_HEIGHT=448 CAMERA_V2_MICRO_BATCH=2"
        )

    detection._yolo_worker = old_ui_yolo_worker
    detection.SmoothBoxManager = OldUIBoxManager
    detection.CameraDetectionV2._infer_gate_probe = _capture_gate_until_sample
    detection.CameraDetectionV2._inject_boxes_probe = _no_pre_tiler_meta
    detection.CameraDetectionV2._scheduler = _old_ui_scheduler
    _install_overlay(detection)

    print(
        "CAMERA_DETECT_BACKEND selected=old-ui-yolo26m source=ui-aspect-ratio-final@865bfedf "
        "model=YOLO26m input=704x448 conf=0.06 iou=0.50 batch=2 "
        "roi=CAM05-verify+CAM06-augment tracker=exact-old-ui-kalman-byte "
        "freshness=300ms-submit/900ms-result flow=OFF reid=OFF nvtracker=OFF",
        flush=True,
    )
