from __future__ import annotations

"""RF-DETR-S detector backend for the Camera V2 core.

The rest of the live pipeline keeps its existing latest-only DeepStream capture,
NvDCF tracker and OSD contract. This module only replaces the sparse detector
worker while preserving the result schema expected by the tracker scheduler.
"""

import importlib.metadata
import os
import time

import numpy as np


def _person_mask(detections, class_id: np.ndarray) -> np.ndarray:
    """Resolve the person class across RF-DETR checkpoint label layouts.

    Current pretrained COCO checkpoints can expose sparse raw COCO category IDs
    (where person is category 1), while fine-tuned checkpoints can expose zero-
    based class IDs. RF-DETR also publishes class_name in Detections.data, so use
    that semantic label whenever available and only fall back to IDs.
    """
    data = getattr(detections, "data", None)
    if isinstance(data, dict):
        names = data.get("class_name")
        if names is not None:
            names = np.asarray(names).astype(str)
            if len(names) == len(class_id):
                normalized = np.char.lower(np.char.strip(names))
                return normalized == "person"

    # Pretrained COCO: raw sparse IDs use 1 for person. Older/remapped or
    # fine-tuned one-class checkpoints commonly use 0. Accept both only as a
    # compatibility fallback when semantic class names are unavailable.
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


def rfdetr_worker(job_q, result_q) -> None:
    """Spawn-safe CUDA RF-DETR-S worker.

    Input frames arrive from GStreamer as BGR. RF-DETR's NumPy API expects RGB,
    so every selected frame is converted before inference. The worker remains
    process-isolated exactly like the previous detector, keeping CUDA failures out
    of the Qt/DeepStream parent process.
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
        torch.cuda.set_device(0)
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

        # Import here, after the child owns CUDA, to avoid any detector-side CUDA
        # context in the parent process.
        from . import detection as det

        startup_delay = float(os.environ.get("CAMERA_V2_DETECT_STARTUP_DELAY", "3.0"))
        if startup_delay > 0:
            time.sleep(startup_delay)

        # Keep the current 16:9 detector branch. RF-DETR supports a non-square
        # predict shape, and 736x416 is valid for RF-DETR-S's patch/window grid.
        infer_shape = (int(det.INFER_HEIGHT), int(det.INFER_WIDTH))
        threshold = float(det.CONF)
        max_det = int(det.MAX_DET)

        model = RFDETRSmall(device="cuda:0")

        # The target GPU is memory constrained. Avoid JIT/deep-copy optimization
        # during bring-up; eager FP32 is the accuracy-safe baseline on Pascal.
        # Once the real six-camera smoke is clean we can separately benchmark an
        # FP16/TensorRT engine without changing tracker behavior.
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
                # RF-DETR documents NumPy input as RGB. The inference side branch
                # delivers BGR from BGRx, so convert without touching display data.
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
                        f"RF-DETR batch mismatch: predictions={len(predictions)} cameras={len(job['cameras'])}"
                    )

                output = {
                    cid: _person_rows(prediction, max_det)
                    for cid, prediction in zip(job["cameras"], predictions)
                }
                result_q.put(
                    {
                        "type": "result",
                        "backend": "RF-DETR-S",
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


def install() -> None:
    """Make RF-DETR-S the detector worker for the active Camera V2 runtime."""
    from . import detection

    detection._yolo_worker = rfdetr_worker
