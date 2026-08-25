from __future__ import annotations

import signal

from .yolo_trt86_shm_bridge import yolo_trt86_shm_worker


def yolo_trt86_fresh_worker(job_q, result_q) -> None:
    """Run the existing SHM bridge, but let only the parent own Ctrl+C.

    Terminal SIGINT is delivered to the whole foreground process group.  The
    detector process must not die in the middle of SharedMemory cleanup; the
    parent Camera V2 runtime handles SIGINT and later sends the normal queue
    sentinel.  Ignored SIGINT is inherited by the TensorRT sidecar as well.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except Exception:
        pass
    yolo_trt86_shm_worker(job_q, result_q)
