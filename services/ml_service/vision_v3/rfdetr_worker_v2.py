from __future__ import annotations

"""RF-DETR-S worker copied from the previously proven camera_v2 backend.

This intentionally avoids the later low-threshold/ROI/hard-mask experiment.  The
working branch used semantic person filtering, 672x384 inference, threshold 0.18,
and one isolated CUDA worker.  Keep that behavior byte-for-byte in spirit while
preserving the Vision V3 queue/result contract.
"""

import importlib.metadata
import os
import time

import numpy as np


def _person_mask(detections, class_id: np.ndarray) -> np.ndarray:
    """Resolve person robustly across RF-DETR checkpoint label layouts.

    Pretrained COCO checkpoints may expose sparse raw category IDs, while other
    checkpoints/remaps can expose zero-based IDs.  The old working runtime first
    trusted RF-DETR's semantic ``class_name`` data and only fell back to IDs.
    """
    data = getattr(detections, "data", None)
    if isinstance(data, dict):
        names = data.get("class_name")
        if names is not None:
            names = np.asarray(names).astype(str)
            if len(names) == len(class_id):
                normalized = np.char.lower(np.char.strip(names))
                return normalized == "person"

    # Compatibility fallback used by the old production branch.
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


def rfdetr_worker_v2(job_q, result_q, cfg: dict) -> None:
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

        startup_delay = float(cfg.get("startup_delay_sec", 0.0))
        if startup_delay > 0.0:
            time.sleep(startup_delay)

        infer_shape = (
            int(cfg.get("capture_height", 384)),
            int(cfg.get("capture_width", 672)),
        )
        threshold = float(cfg.get("threshold", 0.18))
        max_det = max(1, int(cfg.get("max_det", 40)))

        model = RFDETRSmall(device=device)
        if bool(cfg.get("optimize_for_inference", False)):
            model.optimize_for_inference()

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
                "device": torch.cuda.get_device_name(torch.cuda.current_device()),
                "cuda": str(torch.version.cuda),
                "model": "RFDETRSmall",
                "version": version,
                "shape": infer_shape,
                "threshold": threshold,
                "policy": "proven-rfdetr-s-final",
            }
        )

        while True:
            job = job_q.get()
            if job is None:
                return

            started = time.monotonic()
            try:
                # Inference branch supplies BGR from BGRx. RF-DETR NumPy input is RGB.
                rgb_frames = [
                    np.ascontiguousarray(frame[..., ::-1])
                    for frame in job["frames"]
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
                        "RF-DETR batch mismatch: "
                        f"predictions={len(predictions)} cameras={len(job['cameras'])}"
                    )

                output = {
                    camera_id: _person_rows(prediction, max_det)
                    for camera_id, prediction in zip(job["cameras"], predictions)
                }
                result_q.put(
                    {
                        "type": "result",
                        "cameras": list(job["cameras"]),
                        "captured": list(job["captured"]),
                        "boxes": output,
                        "batch_ms": (ended - started) * 1000.0,
                        "roi_inputs": 0,
                        "roi_variants": 0,
                        "hard_rejects": 0,
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
