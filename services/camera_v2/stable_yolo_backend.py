from __future__ import annotations

"""Old-stable YOLO26m person-only backend on the proven Pascal camera wall.

The capture tile intentionally stays close to the 16:9 CCTV source. Ultralytics
owns the final letterbox to the old stable 448x704 model input, exactly as in the
stable service path. Predictions therefore come back in capture-tile coordinates
and CameraDetectionV2's existing source-space scaling remains correct.
"""

import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _resolve_model(spec: str) -> str:
    p = Path(spec).expanduser()
    if p.is_file():
        return str(p)
    root = Path(__file__).resolve().parents[2]
    local = root / spec
    return str(local) if local.is_file() else spec


def stable_yolo_worker(job_q, result_q) -> None:
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

        startup_delay = float(os.environ.get("CAMERA_V2_DETECT_STARTUP_DELAY", "2.0"))
        if startup_delay > 0:
            time.sleep(startup_delay)

        model_spec = os.environ.get("CAMERA_V2_YOLO_MODEL", "yolo26m.pt")
        model_path = _resolve_model(model_spec)
        model = YOLO(model_path)

        names = dict(model.names or {})
        person_name = str(names.get(0, "")).strip().lower()
        if person_name != "person":
            raise RuntimeError(
                f"YOLO person-class contract failed: model.names[0]={names.get(0)!r}"
            )

        model_w = int(os.environ.get("CAMERA_V2_YOLO_IMGSZ_WIDTH", "704"))
        model_h = int(os.environ.get("CAMERA_V2_YOLO_IMGSZ_HEIGHT", "448"))
        threshold = float(det.CONF)
        iou = float(det.IOU)
        max_det = int(det.MAX_DET)

        kwargs = {
            "imgsz": (model_h, model_w),
            "classes": [0],
            "conf": threshold,
            "iou": iou,
            "max_det": max_det,
            "device": "cuda:0",
            "verbose": False,
            "stream": False,
        }

        warm = [
            np.zeros((int(det.INFER_HEIGHT), int(det.INFER_WIDTH), 3), dtype=np.uint8)
            for _ in range(int(det.MICRO_BATCH))
        ]
        with torch.inference_mode():
            model.predict(source=warm, **kwargs)

        print(
            "STABLE_YOLO_READY "
            f"model={model_path} device={torch.cuda.get_device_name(0)} "
            f"capture={det.INFER_WIDTH}x{det.INFER_HEIGHT} "
            f"imgsz={model_w}x{model_h} classes=[0:person] "
            f"conf={threshold:.2f} iou={iou:.2f} batch={det.MICRO_BATCH}",
            flush=True,
        )
        result_q.put(
            {
                "type": "ready",
                "backend": "YOLO26m-person-only",
                "device": torch.cuda.get_device_name(0),
                "cuda": str(torch.version.cuda),
                "model": model_path,
                "capture_shape": (int(det.INFER_HEIGHT), int(det.INFER_WIDTH)),
                "model_shape": (model_h, model_w),
                "threshold": threshold,
            }
        )

        telemetry_budget = max(
            0, int(os.environ.get("CAMERA_V2_STABLE_YOLO_LOG_BUDGET", "36"))
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
                        f"YOLO batch mismatch: predictions={len(predictions)} "
                        f"cameras={len(job['cameras'])}"
                    )

                output = {}
                max_scores = {}
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
                    max_scores[cid] = best

                batch_ms = (ended - started) * 1000.0
                if telemetry_budget > 0:
                    telemetry_budget -= 1
                    counts = " ".join(
                        f"{cid}:{len(output.get(cid, []))}@{max_scores.get(cid, 0.0):.2f}"
                        for cid in job["cameras"]
                    )
                    print(
                        f"STABLE_YOLO_RESULT batch={batch_ms:.1f}ms persons=[{counts}]",
                        flush=True,
                    )

                result_q.put(
                    {
                        "type": "result",
                        "backend": "YOLO26m-person-only",
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
                result_q.put(
                    {"type": "batch_error", "error": f"YOLO26m {type(exc).__name__}: {exc}"}
                )
    except BaseException as exc:
        result_q.put(
            {"type": "fatal", "error": f"YOLO26m {type(exc).__name__}: {exc}"}
        )


def _capture_gate_until_sample(self, _pad, _info, cid: str):
    with self.capture_lock:
        requested = bool(self.capture_requested.get(cid, False))
    return self.Gst.PadProbeReturn.OK if requested else self.Gst.PadProbeReturn.DROP


def install() -> None:
    from . import detection
    from .stable_visual_adapter import StableVisualFlowBoxManager

    class PascalStableVisualFlowBoxManager(StableVisualFlowBoxManager):
        """Expose legacy counter fields expected by CameraPascalSafeRuntime stats."""

        @property
        def max_age(self):
            return self.flow_hard_age_sec

        @property
        def tracks(self):
            # Stats only: map exact old tracker state into the tiny legacy shape
            # used by _active_motion_counts. This does not alter tracking truth.
            output = {}
            with self.lock:
                for cid, tracker in self._trackers.items():
                    with tracker._lock:
                        output[cid] = {
                            int(tid): SimpleNamespace(
                                last_det_t=float(track.last_observation)
                            )
                            for tid, track in tracker._tracks.items()
                            if track.hits >= tracker.strong_confirm_hits
                        }
            return output

    detection._yolo_worker = stable_yolo_worker
    detection.SmoothBoxManager = PascalStableVisualFlowBoxManager
    detection.CameraDetectionV2._infer_gate_probe = _capture_gate_until_sample

    pascal_safe = os.environ.get("CAMERA_V2_PASCAL_SAFE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if pascal_safe and not getattr(
        detection.CameraDetectionV2, "_camera_v2_stable_flow_init_wrapped", False
    ):
        original_init = detection.CameraDetectionV2.__init__

        def _init_with_stable_motion_flow(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            from .motion_flow_branch import attach_motion_flow

            attach_motion_flow(self)

        detection.CameraDetectionV2.__init__ = _init_with_stable_motion_flow
        detection.CameraDetectionV2._camera_v2_stable_flow_init_wrapped = True

    print(
        "CAMERA_DETECT_BACKEND selected=stable-yolo26m-v2 "
        "worker=person-only classes[0] preprocess=aspect-preserving "
        "tracker=old-stable-adaptive-kalman-byte flow=continuous-lk",
        flush=True,
    )
