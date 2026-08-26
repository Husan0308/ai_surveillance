from __future__ import annotations

import os
import signal
from pathlib import Path


def yolo26m_trt86_rescue_worker(job_q, result_q) -> None:
    """CAM-05-only YOLO26m rescue on the proven 672x384 TRT8.6 SHM bridge.

    The SHM bridge resolves its Python, worker, engine and confidence environment
    inside yolo_trt86_shm_worker(), not at module import time. Therefore these
    child-local environment overrides are sufficient even under multiprocessing
    spawn. No second GStreamer branch or resize is created.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    root = Path(__file__).resolve().parents[2]
    engine = os.environ.get(
        "CAMERA_V2_RESCUE_TRT86_ENGINE",
        str(root / "artifacts/yolo26m_trt86/yolo26m-672x384-b1-fp32-trt86.engine"),
    )
    worker = os.environ.get(
        "CAMERA_V2_RESCUE_TRT86_SHM_WORKER",
        str(root / "scripts/yolo26_trt86_shm_worker_v3.py"),
    )
    conf = os.environ.get("CAMERA_V2_RESCUE_CONF", "0.08")
    os.environ["CAMERA_V2_TRT86_ENGINE"] = engine
    os.environ["CAMERA_V2_TRT86_SHM_WORKER"] = worker
    os.environ["CAMERA_V2_DETECT_CONF"] = conf
    os.environ["CAMERA_V2_MAX_DET"] = os.environ.get(
        "CAMERA_V2_RESCUE_MAX_DET", "40"
    )
    os.environ.pop("CAMERA_V2_PARITY_CAPTURE_CAMERAS", None)
    os.environ.pop("CAMERA_V2_PARITY_SAMPLES_PER_CAMERA", None)
    os.environ.pop("CAMERA_V2_PARITY_DIR", None)

    print(
        "CAMERA_RESCUE_BIND "
        f"engine={Path(engine).name} worker={Path(worker).name} conf={float(conf):.2f}",
        flush=True,
    )

    from .yolo_trt86_shm_bridge import yolo_trt86_shm_worker

    yolo_trt86_shm_worker(job_q, result_q)
