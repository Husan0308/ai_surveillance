from __future__ import annotations

"""RF-DETR-S detector backend for Camera V2.

The detector worker is process-isolated and preserves the existing CameraDetectionV2
job/result schema. Tracker selection is intentionally outside this module: the GTX
1050 Ti production controller constructs CameraPascalSafeRuntime directly, while
other runtimes may opt into the sparse NvDCF contract.
"""

import importlib.metadata
import os
import time

import numpy as np


def _person_mask(detections, class_id: np.ndarray) -> np.ndarray:
    """Resolve the person class across RF-DETR checkpoint label layouts."""

    data = getattr(detections, "data", None)
    if isinstance(data, dict):
        names = data.get("class_name")
        if names is not None:
            names = np.asarray(names).astype(str)
            if len(names) == len(class_id):
                normalized = np.char.lower(np.char.strip(names))
                return normalized == "person"

    # Pretrained COCO checkpoints may expose sparse category id 1 for person;
    # older/remapped or one-class checkpoints may expose zero-based id 0.
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
    """Spawn-safe CUDA RF-DETR-S worker."""

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

        # Import after the child owns CUDA. This keeps the parent Qt/DeepStream
        # process free of detector-side CUDA model state.
        from . import detection as det

        startup_delay = float(os.environ.get("CAMERA_V2_DETECT_STARTUP_DELAY", "3.0"))
        if startup_delay > 0:
            time.sleep(startup_delay)

        infer_shape = (int(det.INFER_HEIGHT), int(det.INFER_WIDTH))
        threshold = float(det.CONF)
        max_det = int(det.MAX_DET)

        model = RFDETRSmall(device="cuda:0")

        # Eager FP32 is the compatibility baseline for the Pascal deployment.
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
                # The GStreamer side branch supplies BGR; RF-DETR NumPy input is RGB.
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


def _capture_gate_until_sample(self, _pad, _info, cid: str):
    """Pass buffers while a capture request is armed; scheduler clears the gate."""

    with self.capture_lock:
        requested = bool(self.capture_requested.get(cid, False))
    if not requested:
        return self.Gst.PadProbeReturn.DROP
    return self.Gst.PadProbeReturn.OK


def install() -> None:
    """Install RF-DETR-S and Pascal-safe anchored + optical-flow tracking."""

    from . import detection
    from .flow_assisted_tracker import FlowAssistedPersonTracker

    detection._yolo_worker = rfdetr_worker
    detection.CameraDetectionV2._infer_gate_probe = _capture_gate_until_sample

    # CameraDetectionV2 resolves SmoothBoxManager from its module globals at
    # runtime.  RF-DETR owns detector corrections; the replacement tracker adds
    # a persistent center anchor and accepts measured 20-FPS optical flow between
    # sparse detector calls.
    detection.SmoothBoxManager = FlowAssistedPersonTracker

    pascal_safe = os.environ.get("CAMERA_V2_PASCAL_SAFE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if pascal_safe:
        # CameraPascalSafeRuntime is defined after this installer runs.  Wrapping
        # its CameraDetectionV2 base initializer lets the normal Pascal virtual
        # _install_osd_and_meta() build the proven display/analysis tee first;
        # only then do we attach a third leaky continuous motion branch.
        if not getattr(detection.CameraDetectionV2, "_camera_v2_flow_init_wrapped", False):
            original_init = detection.CameraDetectionV2.__init__

            def _init_with_motion_flow(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                from .motion_flow_branch import attach_motion_flow

                attach_motion_flow(self)

            detection.CameraDetectionV2.__init__ = _init_with_motion_flow
            detection.CameraDetectionV2._camera_v2_flow_init_wrapped = True
    else:
        # Supported/non-Pascal runtimes may still use the existing sparse external
        # detector contract with NvDCF.  The GTX 1050 Ti production controller
        # never imports that tracker path.
        from .sparse_tracker_contract import install_sparse_tracker_contract

        install_sparse_tracker_contract()
