from __future__ import annotations

"""RF-DETR-S backend on the exact rebuild/gpu-v2-clean detection/tracking stack.

This module deliberately does NOT replace the old detection logic. It changes only
`detection._yolo_worker` to RF-DETR-S. Everything after detector output remains the
proven stack from rebuild/gpu-v2-clean:

fresh-frame ticket capture -> dedup/full-body guard -> detector latency compensation
-> fresh metadata injection -> per-frame NvDCF -> tracker-current OSD -> display-only
padding in higher runtimes.

There is no Pascal-safe motion-predictor fallback here. This branch exists to prove
or disprove the exact old behavior on the deployment machine before anything is
ported back into the clean Vision V3 branch.
"""

import importlib.metadata
import os
import time

import numpy as np


def _person_mask(detections, class_id: np.ndarray) -> np.ndarray:
    data = getattr(detections, "data", None)
    if isinstance(data, dict):
        names = data.get("class_name")
        if names is not None:
            names = np.asarray(names).astype(str)
            if len(names) == len(class_id):
                normalized = np.char.lower(np.char.strip(names))
                return normalized == "person"
    return np.isin(class_id, (0, 1))


def _person_rows(detections, max_det: int) -> list[tuple[list[float], float]]:
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
        rows.append(([float(v) for v in box], float(confidence[int(idx)])))
    return rows


def rfdetr_worker(job_q, result_q) -> None:
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

        startup_delay = float(os.environ.get("CAMERA_V2_DETECT_STARTUP_DELAY", "3.0"))
        if startup_delay > 0:
            time.sleep(startup_delay)

        infer_shape = (int(det.INFER_HEIGHT), int(det.INFER_WIDTH))
        threshold = float(det.CONF)
        max_det = int(det.MAX_DET)
        model = RFDETRSmall(device="cuda:0")

        warm = np.zeros((infer_shape[0], infer_shape[1], 3), dtype=np.uint8)
        with torch.inference_mode():
            model.predict(warm, threshold=threshold, shape=infer_shape, include_source_image=False)

        try:
            version = importlib.metadata.version("rfdetr")
        except Exception:
            version = "unknown"

        result_q.put({
            "type": "ready",
            "backend": "RF-DETR-S",
            "device": torch.cuda.get_device_name(0),
            "cuda": str(torch.version.cuda),
            "model": "RFDETRSmall",
            "version": version,
            "shape": infer_shape,
            "threshold": threshold,
        })

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
                        shape=infer_shape,
                        include_source_image=False,
                    )
                ended = time.monotonic()

                if not isinstance(predictions, (list, tuple)):
                    predictions = [predictions]
                if len(predictions) != len(job["cameras"]):
                    raise RuntimeError(
                        f"RF-DETR batch mismatch: predictions={len(predictions)} cameras={len(job['cameras'])}"
                    )

                output = {
                    cid: _person_rows(prediction, max_det)
                    for cid, prediction in zip(job["cameras"], predictions)
                }
                result_q.put({
                    "type": "result",
                    "backend": "RF-DETR-S",
                    "cameras": job["cameras"],
                    "captured": job["captured"],
                    "boxes": output,
                    "batch_ms": (ended - started) * 1000.0,
                })
            except torch.cuda.OutOfMemoryError as exc:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                result_q.put({"type": "batch_error", "error": f"RF-DETR-S CUDA OOM: {exc}"})
            except BaseException as exc:
                result_q.put({"type": "batch_error", "error": f"RF-DETR-S {type(exc).__name__}: {exc}"})
    except BaseException as exc:
        result_q.put({"type": "fatal", "error": f"RF-DETR-S {type(exc).__name__}: {exc}"})


def main() -> int:
    from . import detection
    detection._yolo_worker = rfdetr_worker

    from .person_tracking_final import CameraPersonTrackingFinal

    print(
        "PROVEN_DETECTION_STACK source=rebuild/gpu-v2-clean "
        f"detector=RF-DETR-S input={detection.INFER_WIDTH}x{detection.INFER_HEIGHT} "
        f"threshold={detection.CONF:.3f} micro_batch={detection.MICRO_BATCH} "
        "dedup=old latency_comp=old nvdcf=old display_smoother=none",
        flush=True,
    )
    return CameraPersonTrackingFinal().run()


if __name__ == "__main__":
    raise SystemExit(main())
