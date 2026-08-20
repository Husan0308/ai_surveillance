from __future__ import annotations

"""RF-DETR-S detector backend for the Pascal-safe Camera V2 wall.

The default production mode is deliberately detector-only until person boxes are
visually proven on the live wall:

    analysis tile -> RF-DETR-S -> strict person filter -> latest raw bbox
                  -> NvDsObjectMeta -> nvmultistreamtiler -> nvdsosd

No temporal tracker, optical flow, ReID or identity logic is installed by the
RF-DETR path.  A short raw-result hold only repeats the newest real detector box
on display frames between sparse round-robin inference calls; it never predicts
motion or creates a box on its own.

The analysis capture remains 16:9 (normally 672x384).  RF-DETR-S performs its
own inference resize at its published 512x512 operating point, and RF-DETR's
postprocessor returns boxes in the original analysis-image pixel coordinates.
"""

import importlib.metadata
import math
import os
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .person_candidate_filter import PersonCandidateFilter


@dataclass(slots=True)
class _RawState:
    rows: list[tuple[float, float, float, float, float]]
    captured: float
    version: int


class RFDETRRawBoxManager:
    """Newest real RF-DETR person boxes only; no prediction or association."""

    def __init__(self, width: int, height: int) -> None:
        self.width = float(width)
        self.height = float(height)
        self.lock = threading.RLock()
        self.max_age = float(os.environ.get("CAMERA_V2_RFDETR_RAW_HOLD_SEC", "2.80"))
        self.side_margin = float(os.environ.get("CAMERA_V2_RFDETR_BOX_SIDE_MARGIN", "0.04"))
        self.top_margin = float(os.environ.get("CAMERA_V2_RFDETR_BOX_TOP_MARGIN", "0.03"))
        self.bottom_margin = float(os.environ.get("CAMERA_V2_RFDETR_BOX_BOTTOM_MARGIN", "0.06"))
        self._states: dict[str, _RawState] = {}
        self._versions: dict[str, int] = {}

    @property
    def tracks(self):
        """Compatibility view for existing Pascal counters; this is not tracking."""
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

    def _guard_box(self, box):
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
        rows: list[tuple[float, float, float, float, float]] = []
        for box, confidence in detections or ():
            try:
                coords = [float(v) for v in box]
                conf = float(confidence)
            except (TypeError, ValueError, OverflowError):
                continue
            if len(coords) != 4 or not all(math.isfinite(v) for v in (*coords, conf)):
                continue
            if conf <= 0.0 or coords[2] <= coords[0] or coords[3] <= coords[1]:
                continue
            x1, y1, x2, y2 = self._guard_box(coords)
            rows.append((x1, y1, x2, y2, min(1.0, conf)))

        with self.lock:
            version = self._versions.get(cid, 0) + 1
            self._versions[cid] = version
            self._states[cid] = _RawState(
                rows=rows,
                captured=float(captured_t),
                version=version,
            )

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


def rfdetr_worker(job_q, result_q) -> None:
    """Spawn-safe CUDA RF-DETR-S worker with strict person-only output."""

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

        startup_delay = float(os.environ.get("CAMERA_V2_DETECT_STARTUP_DELAY", "3.0"))
        if startup_delay > 0:
            time.sleep(startup_delay)

        capture_shape = (int(det.INFER_HEIGHT), int(det.INFER_WIDTH))
        model_shape = (
            int(os.environ.get("CAMERA_V2_RFDETR_MODEL_HEIGHT", "512")),
            int(os.environ.get("CAMERA_V2_RFDETR_MODEL_WIDTH", "512")),
        )
        threshold = float(det.CONF)
        max_det = int(det.MAX_DET)
        person_filter = PersonCandidateFilter()
        telemetry_budget = max(
            0, int(os.environ.get("CAMERA_V2_RFDETR_TRUTH_LOG_BUDGET", "72"))
        )

        model = RFDETRSmall(device="cuda:0")
        warm = np.zeros((capture_shape[0], capture_shape[1], 3), dtype=np.uint8)
        # RF-DETR expects RGB numpy input.  The live appsink is BGR, so the live
        # path reverses channels below; a zero warmup frame is channel-invariant.
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
            "RFDETR_TRUTH_READY "
            f"version={version} model=RF-DETR-S device={torch.cuda.get_device_name(0)} "
            f"capture={capture_shape[1]}x{capture_shape[0]} "
            f"model_shape={model_shape[1]}x{model_shape[0]} "
            f"threshold={threshold:.2f} person_ids={person_filter.person_ids}",
            flush=True,
        )
        result_q.put(
            {
                "type": "ready",
                "backend": "RF-DETR-S-truth",
                "device": torch.cuda.get_device_name(0),
                "cuda": str(torch.version.cuda),
                "model": "RFDETRSmall",
                "version": version,
                "capture_shape": capture_shape,
                "model_shape": model_shape,
                "threshold": threshold,
                "person_class_ids": person_filter.person_ids,
            }
        )

        while True:
            job = job_q.get()
            if job is None:
                return

            started = time.monotonic()
            try:
                rgb_frames = [
                    np.ascontiguousarray(frame[..., ::-1]) for frame in job["frames"]
                ]
                with torch.inference_mode():
                    predictions = model.predict(
                        rgb_frames,
                        threshold=threshold,
                        shape=model_shape,
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
                filter_stats = {}
                summary = []
                for cid, prediction in zip(job["cameras"], predictions):
                    rows, stats = person_filter.filter(prediction, max_det)
                    output[cid] = rows
                    best = max((float(score) for _box, score in rows), default=0.0)
                    summary.append(f"{cid}:{len(rows)}@{best:.2f}")
                    filter_stats[cid] = {
                        "raw": stats.raw,
                        "class_rejected": stats.class_rejected,
                        "geometry_rejected": stats.geometry_rejected,
                        "duplicate_rejected": stats.duplicate_rejected,
                        "kept": stats.kept,
                        "class_mode": stats.class_mode,
                        "raw_ids": stats.raw_ids,
                        "raw_names": stats.raw_names,
                    }
                    if telemetry_budget > 0:
                        names = ",".join(stats.raw_names) if stats.raw_names else "-"
                        ids = ",".join(str(v) for v in stats.raw_ids) if stats.raw_ids else "-"
                        print(
                            "RFDETR_TRUTH_FILTER "
                            f"camera={cid} mode={stats.class_mode} raw={stats.raw} "
                            f"class_reject={stats.class_rejected} "
                            f"geom_reject={stats.geometry_rejected} "
                            f"dedup_reject={stats.duplicate_rejected} kept={stats.kept} "
                            f"raw_ids=[{ids}] raw_names=[{names}]",
                            flush=True,
                        )
                if telemetry_budget > 0:
                    telemetry_budget -= 1
                    print(
                        f"RFDETR_TRUTH_RESULT batch={(ended-started)*1000.0:.1f}ms "
                        f"persons=[{' '.join(summary)}]",
                        flush=True,
                    )

                result_q.put(
                    {
                        "type": "result",
                        "backend": "RF-DETR-S-truth",
                        "cameras": job["cameras"],
                        "captured": job["captured"],
                        "boxes": output,
                        "filter_stats": filter_stats,
                        "batch_ms": (ended - started) * 1000.0,
                    }
                )
            except torch.cuda.OutOfMemoryError as exc:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                result_q.put(
                    {"type": "batch_error", "error": f"RF-DETR-S CUDA OOM: {exc}"}
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
            {"type": "fatal", "error": f"RF-DETR-S {type(exc).__name__}: {exc}"}
        )


def _capture_gate_until_sample(self, _pad, _info, cid: str):
    with self.capture_lock:
        requested = bool(self.capture_requested.get(cid, False))
    return self.Gst.PadProbeReturn.OK if requested else self.Gst.PadProbeReturn.DROP


def _inject_rfdetr_truth_probe(self, _pad, info):
    """Attach current real RF-DETR boxes to every mux batch for OSD rendering."""
    buffer = info.get_buffer()
    if buffer is None:
        return self.Gst.PadProbeReturn.OK

    now = time.monotonic()
    added = 0
    logged = getattr(self, "_rfdetr_truth_logged_versions", None)
    if logged is None:
        logged = {}
        self._rfdetr_truth_logged_versions = logged

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
                "RFDETR_TRUTH_META "
                f"camera={cid} source_id={source_id} version={version} "
                f"raw_boxes={len(rows)} injected={result} age={age_ms:.1f}ms",
                flush=True,
            )

    with self.det_lock:
        self.meta_boxes += added
    return self.Gst.PadProbeReturn.OK


def install() -> None:
    """Install the selected detector backend before CameraDetectionV2 is built."""

    selected = os.environ.get("CAMERA_V2_DETECT_BACKEND", "rfdetr-s").strip().lower()
    if selected in {"stable-yolo26m", "yolo26m", "yolo", "stable-yolo"}:
        # Kept only as an explicit diagnostic compatibility mode.  Production
        # launcher selects RF-DETR-S and never enters this branch.
        from .stable_yolo_backend import install as install_stable_yolo

        install_stable_yolo()
        return

    if selected not in {"rfdetr-s", "rfdetr", "rf-detr-s", ""}:
        raise RuntimeError(f"unsupported CAMERA_V2_DETECT_BACKEND={selected!r}")

    from . import detection

    detection._yolo_worker = rfdetr_worker
    detection.SmoothBoxManager = RFDETRRawBoxManager
    detection.CameraDetectionV2._infer_gate_probe = _capture_gate_until_sample
    detection.CameraDetectionV2._inject_boxes_probe = _inject_rfdetr_truth_probe

    print(
        "CAMERA_DETECT_BACKEND selected=rfdetr-s-truth "
        "model=RF-DETR-S tracker=OFF flow=OFF reid=OFF "
        "path=analysis-tile->person-filter->raw-meta->tiler->osd",
        flush=True,
    )
