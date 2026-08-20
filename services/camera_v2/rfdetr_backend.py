from __future__ import annotations

"""Selectable detector backend for Camera V2.

RF-DETR-S remains the production comparison path.  Setting
``CAMERA_V2_DETECT_BACKEND=stable-yolo26m`` installs the restored old-stable
YOLO26m person-only detector plus the exact adaptive-Kalman/Byte visual tracker
core, while keeping the proven Pascal DeepStream camera/display graph intact.
"""

import importlib.metadata
import os
import time

import numpy as np

from .person_candidate_filter import PersonCandidateFilter


def rfdetr_worker(job_q, result_q) -> None:
    """Spawn-safe CUDA RF-DETR-S worker with strict person-only filtering."""

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

        infer_shape = (int(det.INFER_HEIGHT), int(det.INFER_WIDTH))
        threshold = float(det.CONF)
        max_det = int(det.MAX_DET)
        person_filter = PersonCandidateFilter()
        telemetry_budget = max(
            0, int(os.environ.get("CAMERA_V2_RFDETR_FILTER_LOG_BUDGET", "18"))
        )

        model = RFDETRSmall(device="cuda:0")
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
                filter_stats = {}
                for cid, prediction in zip(job["cameras"], predictions):
                    rows, stats = person_filter.filter(prediction, max_det)
                    output[cid] = rows
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
                        telemetry_budget -= 1
                        names = ",".join(stats.raw_names) if stats.raw_names else "-"
                        ids = ",".join(str(v) for v in stats.raw_ids) if stats.raw_ids else "-"
                        print(
                            "CAMERA_RFDETR_FILTER "
                            f"camera={cid} class_mode={stats.class_mode} "
                            f"raw={stats.raw} class_reject={stats.class_rejected} "
                            f"geom_reject={stats.geometry_rejected} "
                            f"dedup_reject={stats.duplicate_rejected} kept={stats.kept} "
                            f"raw_ids=[{ids}] raw_names=[{names}] "
                            f"fallback_person_ids={person_filter.person_ids}",
                            flush=True,
                        )

                result_q.put(
                    {
                        "type": "result",
                        "backend": "RF-DETR-S",
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
    if not requested:
        return self.Gst.PadProbeReturn.DROP
    return self.Gst.PadProbeReturn.OK


def install() -> None:
    """Install the selected detector/tracker backend before CameraDetectionV2 init."""

    selected = os.environ.get("CAMERA_V2_DETECT_BACKEND", "rfdetr-s").strip().lower()
    if selected in {"stable-yolo26m", "yolo26m", "yolo", "stable-yolo"}:
        from .stable_yolo_backend import install as install_stable_yolo

        install_stable_yolo()
        return

    from . import detection
    from .flow_assisted_tracker import FlowAssistedPersonTracker

    detection._yolo_worker = rfdetr_worker
    detection.CameraDetectionV2._infer_gate_probe = _capture_gate_until_sample
    detection.SmoothBoxManager = FlowAssistedPersonTracker

    pascal_safe = os.environ.get("CAMERA_V2_PASCAL_SAFE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if pascal_safe:
        if not getattr(detection.CameraDetectionV2, "_camera_v2_flow_init_wrapped", False):
            original_init = detection.CameraDetectionV2.__init__

            def _init_with_motion_flow(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                from .motion_flow_branch import attach_motion_flow

                attach_motion_flow(self)

            detection.CameraDetectionV2.__init__ = _init_with_motion_flow
            detection.CameraDetectionV2._camera_v2_flow_init_wrapped = True
    else:
        from .sparse_tracker_contract import install_sparse_tracker_contract

        install_sparse_tracker_contract()
