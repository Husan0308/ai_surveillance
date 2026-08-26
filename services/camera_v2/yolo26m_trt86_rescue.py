from __future__ import annotations

import importlib
import os
import signal
from pathlib import Path


def yolo26m_trt86_rescue_worker(job_q, result_q) -> None:
    """CAM-05-only YOLO26m rescue using the proven 672x384 TRT8.6 SHM bridge.

    Environment overrides are process-local: the primary YOLO26s worker keeps its
    own engine and confidence. No additional GStreamer branch or resize is needed;
    the rescue consumes the exact same 672x384 BGR frame captured for primary.

    With multiprocessing spawn, the child imports the application's module graph
    before entering this target. That graph may already have imported the primary
    bridge and cached its S-engine constants. Reload the bridge after applying the
    rescue environment so this process is provably bound to YOLO26m.
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
    os.environ.pop("CAMERA_V2_PARITY_CAPTURE_CAMERAS", None)
    os.environ.pop("CAMERA_V2_PARITY_SAMPLES_PER_CAMERA", None)
    os.environ.pop("CAMERA_V2_PARITY_DIR", None)

    from . import yolo_trt86_shm_bridge as bridge

    bridge = importlib.reload(bridge)
    print(
        "CAMERA_RESCUE_BIND "
        f"engine={Path(bridge.TRT_ENGINE).name} "
        f"worker={Path(bridge.TRT_WORKER).name} conf={bridge.DETECT_CONF:.2f}",
        flush=True,
    )
    bridge.yolo_trt86_shm_worker(job_q, result_q)
