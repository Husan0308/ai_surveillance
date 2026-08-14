from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

from services.ml_service.pose.coordinator import PoseKeypoint, PosePerson, PoseResult

from .detector import (
    DetectionResult,
    PersonBox,
    YoloDetectorWorker,
    _box_center_inside,
    _deduplicate_boxes,
    _filter_hard_masks,
    _map_full_boxes,
    _map_roi_boxes,
    _predict_kwargs,
    _roi_predict_kwargs,
    _rotate_image,
)

log = logging.getLogger(__name__)


class LatestPoseStore:
    """Exactly one newest pose result per camera."""

    def __init__(self):
        self._lock = threading.Lock()
        self._results: dict[str, PoseResult] = {}

    def put(self, result: PoseResult) -> None:
        with self._lock:
            previous = self._results.get(result.camera_id)
            if previous is None or result.frame_id >= previous.frame_id:
                self._results[result.camera_id] = result

    def snapshot(self):
        with self._lock:
            return dict(self._results)


class UnifiedPoseProvider:
    """Read-only pose provider backed by the detector CUDA worker."""

    def __init__(self, owner: "UnifiedYoloDetectorWorker"):
        self._owner = owner

    def snapshot(self):
        return self._owner.pose_results.snapshot()

    def metrics(self):
        return self._owner.pose_metrics()


def _pose_crop(entry: dict, box):
    """Crop a source-space detector box from the resized detector input."""
    source_w = max(1.0, float(entry["source_w"]))
    source_h = max(1.0, float(entry["source_h"]))
    resized_h, resized_w = entry["full_shape"]
    sx = float(resized_w) / source_w
    sy = float(resized_h) / source_h

    x1 = max(0, min(resized_w - 1, int(float(box[0]) * sx)))
    y1 = max(0, min(resized_h - 1, int(float(box[1]) * sy)))
    x2 = max(x1 + 1, min(resized_w, int(round(float(box[2]) * sx))))
    y2 = max(y1 + 1, min(resized_h, int(round(float(box[3]) * sy))))
    crop = entry["full_image"][y1:y2, x1:x2]
    return crop, (x1, y1), (source_w / float(resized_w), source_h / float(resized_h))


def _raw_pose_person(prediction, source_box, offset, back_scale):
    keypoints = getattr(prediction, "keypoints", None)
    if keypoints is None or getattr(keypoints, "xy", None) is None or len(keypoints.xy) == 0:
        return None

    index = 0
    pred_boxes = getattr(prediction, "boxes", None)
    if pred_boxes is not None and getattr(pred_boxes, "conf", None) is not None and len(pred_boxes.conf):
        try:
            index = int(pred_boxes.conf.argmax().item())
        except Exception:
            index = 0
    if index >= len(keypoints.xy):
        index = 0

    xy = keypoints.xy[index].detach().cpu().tolist()
    conf_tensor = getattr(keypoints, "conf", None)
    if conf_tensor is not None and len(conf_tensor) > index:
        confs = conf_tensor[index].detach().cpu().tolist()
    else:
        confs = [1.0] * len(xy)

    ox, oy = offset
    back_x, back_y = back_scale
    points = [
        (float(p[0] + ox) * back_x, float(p[1] + oy) * back_y, float(c))
        for p, c in zip(xy, confs)
    ]

    confidence = float(source_box[4])
    if pred_boxes is not None and getattr(pred_boxes, "conf", None) is not None and len(pred_boxes.conf) > index:
        try:
            confidence = float(pred_boxes.conf[index].detach().cpu().item())
        except Exception:
            pass

    return {
        "bbox": tuple(float(v) for v in source_box[:4]),
        "confidence": confidence,
        "keypoints": points,
    }


def _unified_detector_process_main(
    input_queue,
    output_queue,
    detector_config: dict,
    model_path: str,
    pose_config: dict,
):
    """Run detector and sparse pose in one process and one CUDA context."""
    import faulthandler

    faulthandler.enable(all_threads=True)
    try:
        import numpy as np
        import torch
        from ultralytics import YOLO

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        kwargs = _predict_kwargs(detector_config)
        roi_kwargs = _roi_predict_kwargs(detector_config, kwargs)
        device = str(kwargs["device"])
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

        detector_model = YOLO(model_path)
        imgsz = kwargs["imgsz"]
        warm_h, warm_w = (
            (int(imgsz[0]), int(imgsz[1]))
            if isinstance(imgsz, tuple)
            else (int(imgsz), int(imgsz))
        )
        detector_model.predict(
            source=[np.zeros((warm_h, warm_w, 3), dtype=np.uint8)],
            **kwargs,
        )
    except BaseException as exc:
        try:
            output_queue.put(("startup_error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
        return

    pose_enabled = bool(pose_config.get("enabled", False))
    pose_model = None
    pose_device_cfg = str(pose_config.get("device", "inherit"))
    pose_device = device if pose_device_cfg in {"", "inherit", "same", "detector"} else pose_device_cfg
    if pose_device != device:
        output_queue.put(
            (
                "pose_startup_error",
                {
                    "error": (
                        f"pose device {pose_device!r} differs from detector device {device!r}; "
                        "unified worker requires the same device"
                    )
                },
            )
        )
        pose_enabled = False

    pose_model_name = str(pose_config.get("model", "yolo26m-pose.pt"))
    pose_imgsz = max(128, int(pose_config.get("imgsz", 256)))
    pose_conf = float(pose_config.get("conf", 0.25))
    pose_every_n = max(1, int(pose_config.get("every_n", 8)))
    pose_max_people = max(1, int(pose_config.get("max_people", 2)))
    pose_max_cameras = max(1, int(pose_config.get("max_cameras_per_batch", 1)))
    pose_max_age_ms = max(0.0, float(pose_config.get("max_frame_age_ms", 850)))
    pose_half = bool(pose_config.get("half", device.startswith("cuda")))
    pose_max_detector_ms = max(
        0.0, float(pose_config.get("max_detector_ms_before_pose", 350))
    )
    pose_seen: dict[str, int] = {}

    if pose_enabled:
        try:
            pose_model = YOLO(pose_model_name)
            pose_model.predict(
                source=[np.zeros((pose_imgsz, pose_imgsz, 3), dtype=np.uint8)],
                imgsz=pose_imgsz,
                conf=pose_conf,
                device=pose_device,
                half=pose_half,
                max_det=1,
                verbose=False,
            )
            output_queue.put(
                (
                    "pose_ready",
                    {
                        "model": pose_model_name,
                        "device": pose_device,
                        "imgsz": pose_imgsz,
                        "half": pose_half,
                    },
                )
            )
        except BaseException as exc:
            try:
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception:
                pass
            pose_model = None
            output_queue.put(
                ("pose_startup_error", {"error": f"{type(exc).__name__}: {exc}"})
            )

    output_queue.put(
        (
            "ready",
            {
                "device": device,
                "model": model_path,
                "pose_enabled": bool(pose_model is not None),
            },
        )
    )

    duplicate_iou = float(detector_config.get("duplicate_iou", 0.58))
    fusion_containment = float(detector_config.get("fusion_containment", 0.84))
    fusion_center_distance = float(detector_config.get("fusion_center_distance", 0.40))
    hard_cfg = dict(detector_config.get("hard_exclusion") or {})
    roi_cfg = dict(detector_config.get("roi_second_pass") or {})
    roi_enabled = bool(roi_cfg.get("enabled", False))
    global_trigger_max = max(0, int(roi_cfg.get("trigger_max_full_roi_persons", 1)))

    while True:
        try:
            payload = input_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        except (EOFError, OSError):
            return
        if payload is None:
            return

        batch_id, entries = payload
        started = time.perf_counter()
        try:
            full_predictions = detector_model.predict(
                source=[entry["full_image"] for entry in entries],
                **kwargs,
            )
        except BaseException as exc:
            try:
                output_queue.put(
                    (
                        "batch_error",
                        {
                            "batch_id": batch_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                )
            except Exception:
                pass
            continue

        boxes_by_camera = {}
        hard_rejects = 0
        for entry, prediction in zip(entries, full_predictions):
            cid = entry["camera_id"]
            boxes = _map_full_boxes(
                prediction,
                entry["source_w"],
                entry["source_h"],
                entry["full_shape"],
            )
            boxes, rejected = _filter_hard_masks(
                cid,
                boxes,
                entry["source_w"],
                entry["source_h"],
                hard_cfg,
            )
            boxes_by_camera[cid] = boxes
            hard_rejects += rejected

        roi_jobs = []
        if roi_enabled:
            for entry in entries:
                roi = entry.get("roi")
                if not roi:
                    continue
                full_inside = sum(
                    1
                    for box in boxes_by_camera.get(entry["camera_id"], ())
                    if _box_center_inside(box, roi["bounds"])
                )
                mode = str(roi.get("mode", "augment")).lower()
                trigger_max = int(
                    roi.get("trigger_max_full_roi_persons", global_trigger_max)
                )
                should_run = (
                    bool(roi.get("always_run", False))
                    or mode == "verify"
                    or full_inside <= trigger_max
                )
                if not should_run:
                    continue
                for rotation in roi.get("rotations", [0]):
                    rotation = int(rotation) % 360
                    roi_jobs.append(
                        (entry, roi, rotation, _rotate_image(roi["image"], rotation))
                    )

        roi_wall_ms = 0.0
        if roi_jobs:
            roi_started = time.perf_counter()
            try:
                roi_predictions = detector_model.predict(
                    source=[job[3] for job in roi_jobs],
                    **roi_kwargs,
                )
                grouped = {}
                for (entry, roi, rotation, _image), prediction in zip(
                    roi_jobs, roi_predictions
                ):
                    accept_conf = float(
                        roi.get(
                            "accept_conf",
                            roi_cfg.get("accept_conf", roi_kwargs["conf"]),
                        )
                    )
                    mapped = [
                        box
                        for box in _map_roi_boxes(prediction, roi, rotation)
                        if box[4] >= accept_conf
                    ]
                    mapped, rejected = _filter_hard_masks(
                        entry["camera_id"],
                        mapped,
                        entry["source_w"],
                        entry["source_h"],
                        hard_cfg,
                    )
                    hard_rejects += rejected
                    grouped.setdefault(entry["camera_id"], []).extend(mapped)

                for entry in entries:
                    roi = entry.get("roi")
                    if not roi:
                        continue
                    cid = entry["camera_id"]
                    mode = str(roi.get("mode", "augment")).lower()
                    roi_boxes = grouped.get(cid, [])
                    if mode == "verify":
                        outside = [
                            box
                            for box in boxes_by_camera.get(cid, [])
                            if not _box_center_inside(box, roi["bounds"])
                        ]
                        boxes_by_camera[cid] = outside + roi_boxes
                    else:
                        boxes_by_camera.setdefault(cid, []).extend(roi_boxes)
            except BaseException as exc:
                try:
                    output_queue.put(
                        (
                            "roi_error",
                            {
                                "batch_id": batch_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                    )
                except Exception:
                    pass
            roi_wall_ms = (time.perf_counter() - roi_started) * 1000.0

        produced = time.monotonic()
        result_entries = []
        total_boxes = 0
        final_boxes: dict[str, list] = {}
        for entry in entries:
            cid = entry["camera_id"]
            boxes = _deduplicate_boxes(
                boxes_by_camera.get(cid, []),
                duplicate_iou,
                fusion_containment,
                fusion_center_distance,
            )
            final_boxes[cid] = boxes
            total_boxes += len(boxes)
            result_entries.append(
                (
                    cid,
                    entry["frame_id"],
                    entry["captured_mono"],
                    produced,
                    boxes,
                )
            )

        detector_wall_ms = (time.perf_counter() - started) * 1000.0
        try:
            output_queue.put(
                (
                    "result",
                    {
                        "batch_id": batch_id,
                        "wall_ms": detector_wall_ms,
                        "roi_wall_ms": roi_wall_ms,
                        "roi_inputs": len({job[0]["camera_id"] for job in roi_jobs}),
                        "roi_variants": len(roi_jobs),
                        "hard_rejects": hard_rejects,
                        "inputs": len(entries),
                        "detections": total_boxes,
                        "entries": result_entries,
                    },
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            return

        if pose_model is None:
            continue

        pose_candidates = []
        now = time.monotonic()
        for entry in entries:
            cid = str(entry["camera_id"])
            pose_seen[cid] = pose_seen.get(cid, 0) + 1
            if pose_seen[cid] % pose_every_n:
                continue
            age_ms = max(0.0, (now - float(entry["captured_mono"])) * 1000.0)
            if pose_max_age_ms and age_ms > pose_max_age_ms:
                continue
            boxes = sorted(
                final_boxes.get(cid, ()),
                key=lambda box: float(box[4]),
                reverse=True,
            )[:pose_max_people]
            if boxes:
                pose_candidates.append((entry, boxes))

        if pose_max_detector_ms and detector_wall_ms > pose_max_detector_ms:
            if pose_candidates:
                output_queue.put(
                    (
                        "pose_skip",
                        {
                            "reason": "detector_budget",
                            "count": len(pose_candidates),
                        },
                    )
                )
            continue

        pose_candidates = pose_candidates[:pose_max_cameras]
        if not pose_candidates:
            continue

        pose_crops = []
        pose_meta = []
        for entry, boxes in pose_candidates:
            for box in boxes:
                crop, offset, back_scale = _pose_crop(entry, box)
                if getattr(crop, "size", 0) == 0:
                    continue
                pose_crops.append(crop)
                pose_meta.append((entry, box, offset, back_scale))

        if not pose_crops:
            continue

        pose_started = time.perf_counter()
        try:
            predictions = pose_model.predict(
                source=pose_crops,
                imgsz=pose_imgsz,
                conf=pose_conf,
                device=pose_device,
                half=pose_half,
                max_det=1,
                verbose=False,
            )
            pose_wall_ms = (time.perf_counter() - pose_started) * 1000.0
            grouped_people: dict[str, list] = {}
            entries_by_camera: dict[str, dict] = {}
            for meta, prediction in zip(pose_meta, predictions):
                entry, source_box, offset, back_scale = meta
                cid = str(entry["camera_id"])
                entries_by_camera[cid] = entry
                person = _raw_pose_person(
                    prediction,
                    source_box,
                    offset,
                    back_scale,
                )
                if person is not None:
                    grouped_people.setdefault(cid, []).append(person)

            pose_produced = time.monotonic()
            raw_pose_entries = []
            for cid, entry in entries_by_camera.items():
                raw_pose_entries.append(
                    (
                        cid,
                        int(entry["frame_id"]),
                        float(entry["captured_mono"]),
                        pose_produced,
                        grouped_people.get(cid, []),
                    )
                )
            output_queue.put(
                (
                    "pose_result",
                    {
                        "wall_ms": pose_wall_ms,
                        "inputs": len(pose_crops),
                        "cameras": len(raw_pose_entries),
                        "people": sum(len(item[4]) for item in raw_pose_entries),
                        "entries": raw_pose_entries,
                    },
                )
            )
        except BaseException as exc:
            try:
                output_queue.put(
                    (
                        "pose_error",
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                )
            except Exception:
                pass


class UnifiedYoloDetectorWorker(YoloDetectorWorker):
    """Detector plus sparse pose using one spawned worker/CUDA context."""

    def __init__(
        self,
        frame_stores,
        config: dict,
        project_root: Path,
        pose_config: dict | None = None,
    ):
        super().__init__(frame_stores, config, project_root)
        self.pose_config = dict(pose_config or {})
        self.pose_results = LatestPoseStore()
        self.pose = UnifiedPoseProvider(self)
        self._pose_ready = False
        self._pose_processed = 0
        self._pose_people = 0
        self._pose_inputs = 0
        self._pose_errors = 0
        self._pose_budget_skips = 0
        self._pose_last_inference_ms = 0.0
        self._pose_last_error = ""

    def _spawn_process(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        self._input_queue = self._ctx.Queue(maxsize=1)
        self._output_queue = self._ctx.Queue(maxsize=32)
        self._process = self._ctx.Process(
            target=_unified_detector_process_main,
            name="core-v1-yolo-unified-cuda",
            args=(
                self._input_queue,
                self._output_queue,
                self.config,
                str(self.model_path),
                self.pose_config,
            ),
            daemon=False,
        )
        self._process.start()
        log.info(
            "CORE_V1_UNIFIED_YOLO_PROCESS_STARTED pid=%s start_method=spawn",
            self._process.pid,
        )

    def _drain_outputs(self):
        if self._output_queue is None:
            return
        while True:
            try:
                kind, payload = self._output_queue.get_nowait()
            except queue.Empty:
                return
            except (EOFError, OSError):
                return

            if kind == "ready":
                with self._lock:
                    self._ready = True
                    self._last_error = ""
                log.info(
                    "CORE_V1_YOLO_READY process_pid=%s device=%s model=%s",
                    self._process.pid if self._process else None,
                    payload.get("device"),
                    payload.get("model"),
                )
                continue

            if kind == "pose_ready":
                with self._lock:
                    self._pose_ready = True
                    self._pose_last_error = ""
                log.info(
                    "CORE_V1_POSE_READY shared_process_pid=%s model=%s device=%s",
                    self._process.pid if self._process else None,
                    payload.get("model"),
                    payload.get("device"),
                )
                continue

            if kind == "pose_startup_error":
                with self._lock:
                    self._pose_ready = False
                    self._pose_errors += 1
                    self._pose_last_error = str(payload.get("error") or payload)
                log.error("CORE_V1_POSE_STARTUP_ERROR %s", self._pose_last_error)
                continue

            if kind == "pose_error":
                with self._lock:
                    self._pose_errors += 1
                    self._pose_last_error = str(payload.get("error") or payload)
                log.error("CORE_V1_POSE_ERROR %s", self._pose_last_error)
                continue

            if kind == "pose_skip":
                with self._lock:
                    self._pose_budget_skips += int(payload.get("count", 0))
                continue

            if kind == "pose_result":
                with self._lock:
                    self._pose_processed += int(payload.get("cameras", 0))
                    self._pose_people += int(payload.get("people", 0))
                    self._pose_inputs += int(payload.get("inputs", 0))
                    self._pose_last_inference_ms = float(payload.get("wall_ms", 0.0))
                    self._pose_last_error = ""
                for cid, frame_id, captured, produced, raw_people in payload.get(
                    "entries", ()
                ):
                    people = []
                    for item in raw_people:
                        keypoints = tuple(
                            PoseKeypoint(float(x), float(y), float(conf))
                            for x, y, conf in item.get("keypoints", ())
                        )
                        people.append(
                            PosePerson(
                                tuple(float(v) for v in item["bbox"]),
                                float(item.get("confidence", 0.0)),
                                keypoints,
                            )
                        )
                    self.pose_results.put(
                        PoseResult(
                            str(cid),
                            int(frame_id),
                            float(captured),
                            float(produced),
                            tuple(people),
                        )
                    )
                continue

            if kind in {"startup_error", "batch_error", "roi_error"}:
                error = (
                    payload
                    if isinstance(payload, str)
                    else payload.get("error", str(payload))
                )
                if (
                    kind != "roi_error"
                    and isinstance(payload, dict)
                    and payload.get("batch_id") == self._inflight_batch_id
                ):
                    self._inflight_batch_id = None
                with self._lock:
                    self._last_error = error
                log.error("CORE_V1_YOLO_%s %s", kind.upper(), error)
                continue

            if kind != "result":
                continue

            if payload.get("batch_id") == self._inflight_batch_id:
                self._inflight_batch_id = None
            now = time.monotonic()
            for cid, frame_id, captured_mono, produced_mono, raw_boxes in payload[
                "entries"
            ]:
                finish_age_ms = max(
                    0.0, (now - float(captured_mono)) * 1000.0
                )
                self._last_finish_age_ms = finish_age_ms
                self._finish_age_ms.append(finish_age_ms)
                self._per_camera_finish_age_ms[str(cid)] = finish_age_ms
                if self.max_result_age_ms > 0 and finish_age_ms > self.max_result_age_ms:
                    self._stale_result_drops += 1
                    continue

                boxes = tuple(PersonBox(*map(float, box)) for box in raw_boxes)
                self.results.put(
                    DetectionResult(
                        camera_id=str(cid),
                        frame_id=int(frame_id),
                        frame_captured_monotonic=float(captured_mono),
                        produced_monotonic=float(produced_mono),
                        boxes=boxes,
                    )
                )
                with self._lock:
                    self._per_camera_inputs[str(cid)] += 1
                    self._per_camera_last_frame_id[str(cid)] = int(frame_id)
                    self._per_camera_last_detection_mono[str(cid)] = float(
                        produced_mono
                    )

            with self._lock:
                self._batches += 1
                self._inputs += int(payload["inputs"])
                self._detections += int(payload["detections"])
                self._roi_inputs += int(payload.get("roi_inputs", 0))
                self._roi_variants += int(payload.get("roi_variants", 0))
                self._hard_rejects += int(payload.get("hard_rejects", 0))
                self._last_batch_ms = float(payload["wall_ms"])
                self._last_roi_ms = float(payload.get("roi_wall_ms", 0.0))
                self._batch_ms.append(self._last_batch_ms)
                self._last_error = ""

    def pose_metrics(self):
        process = self._process
        with self._lock:
            return {
                "enabled": bool(self.pose_config.get("enabled", False)),
                "ready": self._pose_ready,
                "shared_cuda_process": True,
                "process_pid": process.pid if process else None,
                "model": str(self.pose_config.get("model", "yolo26m-pose.pt")),
                "device": str(self.pose_config.get("device", "inherit")),
                "detector_device": str(self.config.get("device", "cuda:0")),
                "imgsz": int(self.pose_config.get("imgsz", 256)),
                "half": bool(
                    self.pose_config.get(
                        "half",
                        str(self.config.get("device", "cuda:0")).startswith("cuda"),
                    )
                ),
                "every_n": max(1, int(self.pose_config.get("every_n", 8))),
                "max_people": max(1, int(self.pose_config.get("max_people", 2))),
                "processed": self._pose_processed,
                "pose_inputs": self._pose_inputs,
                "people": self._pose_people,
                "budget_skips": self._pose_budget_skips,
                "errors": self._pose_errors,
                "last_inference_ms": self._pose_last_inference_ms,
                "last_error": self._pose_last_error,
            }

    def metrics(self):
        metrics = super().metrics()
        metrics["pose"] = self.pose_metrics()
        metrics["cuda_topology"] = "single_process_detector_and_pose"
        return metrics
