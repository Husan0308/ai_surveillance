from __future__ import annotations

"""Tracker-free YOLO26m person detection truth path.

This backend intentionally removes every temporal layer from the equation:

    analysis tile -> YOLO26m classes=[0] -> raw source-space boxes -> NvDsObjectMeta -> OSD

It is used to prove and stabilize person detection before any tracker, optical
flow, ReID or identity logic is allowed back into the production path.
"""

import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np


@dataclass(slots=True)
class _RawState:
    rows: list[tuple[float, float, float, float, float]]
    captured: float
    version: int


class RawDetectionBoxManager:
    """Keep only the newest detector truth for each camera; no tracking."""

    def __init__(self, width: int, height: int) -> None:
        self.width = float(width)
        self.height = float(height)
        self.lock = threading.RLock()
        self.max_age = float(os.environ.get("CAMERA_V2_RAW_BOX_MAX_AGE", "2.40"))
        self.side_margin = float(os.environ.get("CAMERA_V2_RAW_BOX_SIDE_MARGIN", "0.05"))
        self.top_margin = float(os.environ.get("CAMERA_V2_RAW_BOX_TOP_MARGIN", "0.03"))
        self.bottom_margin = float(os.environ.get("CAMERA_V2_RAW_BOX_BOTTOM_MARGIN", "0.07"))
        self._states: dict[str, _RawState] = {}
        self._versions: dict[str, int] = {}

    @property
    def tracks(self):
        """Compatibility-only view for Pascal stats; not a tracker."""
        now = time.monotonic()
        output = {}
        with self.lock:
            for cid, state in self._states.items():
                if now - state.captured > self.max_age:
                    output[cid] = {}
                    continue
                output[cid] = {
                    index + 1: SimpleNamespace(last_det_t=state.captured)
                    for index in range(len(state.rows))
                }
        return output

    def _expand(self, box):
        x1, y1, x2, y2 = [float(v) for v in box]
        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)
        x1 -= w * self.side_margin
        x2 += w * self.side_margin
        y1 -= h * self.top_margin
        y2 += h * self.bottom_margin
        x1 = max(0.0, min(self.width - 2.0, x1))
        y1 = max(0.0, min(self.height - 2.0, y1))
        x2 = max(x1 + 1.0, min(self.width - 1.0, x2))
        y2 = max(y1 + 1.0, min(self.height - 1.0, y2))
        return x1, y1, x2, y2

    def update(self, cid: str, captured_t: float, detections) -> None:
        rows = []
        for box, confidence in detections or ():
            try:
                conf = float(confidence)
                coords = [float(v) for v in box]
            except (TypeError, ValueError, OverflowError):
                continue
            if len(coords) != 4 or not all(math.isfinite(v) for v in (*coords, conf)):
                continue
            if conf <= 0.0 or coords[2] <= coords[0] or coords[3] <= coords[1]:
                continue
            x1, y1, x2, y2 = self._expand(coords)
            rows.append((x1, y1, x2, y2, min(1.0, conf)))

        with self.lock:
            version = self._versions.get(cid, 0) + 1
            self._versions[cid] = version
            self._states[cid] = _RawState(rows=rows, captured=float(captured_t), version=version)

    def render(self, cid: str, now: float):
        with self.lock:
            state = self._states.get(cid)
            if state is None or float(now) - state.captured > self.max_age:
                return []
            return list(state.rows)

    def version(self, cid: str) -> int:
        with self.lock:
            state = self._states.get(cid)
            return int(state.version) if state is not None else 0

    def age(self, cid: str, now: float) -> float | None:
        with self.lock:
            state = self._states.get(cid)
            if state is None:
                return None
            return max(0.0, float(now) - state.captured)


def _resolve_model(spec: str) -> str:
    p = Path(spec).expanduser()
    if p.is_file():
        return str(p)
    root = Path(__file__).resolve().parents[2]
    local = root / spec
    return str(local) if local.is_file() else spec


def stable_yolo_truth_worker(job_q, result_q) -> None:
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
        from . import detection as det

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        torch.cuda.set_device(0)
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

        delay = float(os.environ.get("CAMERA_V2_DETECT_STARTUP_DELAY", "1.5"))
        if delay > 0:
            time.sleep(delay)

        model_spec = os.environ.get("CAMERA_V2_YOLO_MODEL", "yolo26m.pt")
        model_path = _resolve_model(model_spec)
        model = YOLO(model_path)
        names = dict(model.names or {})
        if str(names.get(0, "")).strip().lower() != "person":
            raise RuntimeError(f"YOLO class-0 must be person, got {names.get(0)!r}")

        model_w = int(os.environ.get("CAMERA_V2_YOLO_IMGSZ_WIDTH", "704"))
        model_h = int(os.environ.get("CAMERA_V2_YOLO_IMGSZ_HEIGHT", "448"))
        threshold = float(det.CONF)
        iou = float(det.IOU)
        max_det = int(det.MAX_DET)

        # YOLO26 docs recommend one-to-many + NMS when accuracy is preferred.
        kwargs = {
            "imgsz": (model_h, model_w),
            "rect": True,
            "classes": [0],
            "conf": threshold,
            "iou": iou,
            "max_det": max_det,
            "device": "cuda:0",
            "verbose": False,
            "stream": False,
            "end2end": False,
        }

        warm = [
            np.zeros((int(det.INFER_HEIGHT), int(det.INFER_WIDTH), 3), dtype=np.uint8)
            for _ in range(int(det.MICRO_BATCH))
        ]
        with torch.inference_mode():
            model.predict(source=warm, **kwargs)

        print(
            "YOLO_TRUTH_READY "
            f"model={model_path} device={torch.cuda.get_device_name(0)} "
            f"capture={det.INFER_WIDTH}x{det.INFER_HEIGHT} "
            f"imgsz={model_w}x{model_h} classes=[0:person] end2end=0 "
            f"conf={threshold:.2f} iou={iou:.2f} batch={det.MICRO_BATCH}",
            flush=True,
        )
        result_q.put(
            {
                "type": "ready",
                "backend": "YOLO26m-person-truth",
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
                    predictions = model.predict(source=frames, **kwargs)
                ended = time.monotonic()

                if len(predictions) != len(job["cameras"]):
                    raise RuntimeError(
                        f"YOLO batch mismatch predictions={len(predictions)} cameras={len(job['cameras'])}"
                    )

                output = {}
                summary = []
                for cid, prediction in zip(job["cameras"], predictions):
                    boxes = getattr(prediction, "boxes", None)
                    rows = []
                    best = 0.0
                    if boxes is not None and len(boxes):
                        xyxy = boxes.xyxy.detach().float().cpu().tolist()
                        confs = boxes.conf.detach().float().cpu().tolist()
                        clss = boxes.cls.detach().float().cpu().tolist()
                        for coords, score, cls_id in zip(xyxy, confs, clss):
                            if int(round(float(cls_id))) != 0:
                                continue
                            score = float(score)
                            rows.append(([float(v) for v in coords], score))
                            best = max(best, score)
                    output[cid] = rows
                    summary.append(f"{cid}:{len(rows)}@{best:.2f}")

                batch_ms = (ended - started) * 1000.0
                print(
                    f"YOLO_TRUTH_RESULT batch={batch_ms:.1f}ms persons=[{' '.join(summary)}]",
                    flush=True,
                )
                result_q.put(
                    {
                        "type": "result",
                        "backend": "YOLO26m-person-truth",
                        "cameras": job["cameras"],
                        "captured": job["captured"],
                        "boxes": output,
                        "batch_ms": batch_ms,
                    }
                )
            except torch.cuda.OutOfMemoryError as exc:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                result_q.put({"type": "batch_error", "error": f"YOLO26m CUDA OOM: {exc}"})
            except BaseException as exc:
                result_q.put({"type": "batch_error", "error": f"YOLO26m {type(exc).__name__}: {exc}"})
    except BaseException as exc:
        result_q.put({"type": "fatal", "error": f"YOLO26m {type(exc).__name__}: {exc}"})


def _capture_gate_until_sample(self, _pad, _info, cid: str):
    with self.capture_lock:
        requested = bool(self.capture_requested.get(cid, False))
    return self.Gst.PadProbeReturn.OK if requested else self.Gst.PadProbeReturn.DROP


def _inject_truth_boxes_probe(self, _pad, info):
    buffer = info.get_buffer()
    if buffer is None:
        return self.Gst.PadProbeReturn.OK

    now = time.monotonic()
    added = 0
    logged = getattr(self, "_truth_logged_versions", None)
    if logged is None:
        logged = {}
        self._truth_logged_versions = logged

    for cid, source_id in self.camera_index.items():
        rows = self.boxes.render(cid, now)
        result = self.bridge.add_boxes(buffer, source_id, rows) if rows else 0
        if result > 0:
            added += result

        version = self.boxes.version(cid)
        if version > logged.get(cid, 0):
            logged[cid] = version
            age = self.boxes.age(cid, now)
            age_ms = -1.0 if age is None else age * 1000.0
            print(
                "YOLO_TRUTH_META "
                f"camera={cid} source_id={source_id} version={version} "
                f"raw_boxes={len(rows)} injected={result} age={age_ms:.1f}ms",
                flush=True,
            )

    with self.det_lock:
        self.meta_boxes += added
    return self.Gst.PadProbeReturn.OK


def install() -> None:
    from . import detection

    detection._yolo_worker = stable_yolo_truth_worker
    detection.SmoothBoxManager = RawDetectionBoxManager
    detection.CameraDetectionV2._infer_gate_probe = _capture_gate_until_sample
    detection.CameraDetectionV2._inject_boxes_probe = _inject_truth_boxes_probe

    print(
        "CAMERA_DETECT_BACKEND selected=stable-yolo26m-truth "
        "worker=YOLO26m classes[0] end2end=0 tracker=OFF flow=OFF "
        "path=raw-detection-to-deepstream-meta",
        flush=True,
    )
