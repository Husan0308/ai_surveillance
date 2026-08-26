from __future__ import annotations

import os
import signal
from pathlib import Path


def yolo26m_trt86_rescue_worker(job_q, result_q) -> None:
    """CAM-05-only YOLO26m rescue using the proven 672x384 TRT8.6 SHM bridge.

    Environment overrides are process-local: the primary YOLO26s worker keeps its
    own engine and confidence. No additional GStreamer branch or resize is needed;
    the rescue consumes the exact same 672x384 BGR frame captured for primary.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    root = Path(__file__).resolve().parents[2]
    os.environ["CAMERA_V2_TRT86_ENGINE"] = os.environ.get(
        "CAMERA_V2_RESCUE_TRT86_ENGINE",
        str(root / "artifacts/yolo26m_trt86/yolo26m-672x384-b1-fp32-trt86.engine"),
    )
    os.environ["CAMERA_V2_TRT86_SHM_WORKER"] = os.environ.get(
        "CAMERA_V2_RESCUE_TRT86_SHM_WORKER",
        str(root / "scripts/yolo26_trt86_shm_worker_v3.py"),
    )
    os.environ["CAMERA_V2_DETECT_CONF"] = os.environ.get(
        "CAMERA_V2_RESCUE_CONF", "0.08"
    )
    os.environ["CAMERA_V2_MAX_DET"] = os.environ.get(
        "CAMERA_V2_RESCUE_MAX_DET", "40"
    )
    # Diagnostic parity capture belongs to primary only; never duplicate it here.
    os.environ.pop("CAMERA_V2_PARITY_CAPTURE_CAMERAS", None)
    os.environ.pop("CAMERA_V2_PARITY_SAMPLES_PER_CAMERA", None)
    os.environ.pop("CAMERA_V2_PARITY_DIR", None)

    from .yolo_trt86_shm_bridge import yolo_trt86_shm_worker

    yolo_trt86_shm_worker(job_q, result_q)
