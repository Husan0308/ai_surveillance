from __future__ import annotations

import signal

from .step2_production_fp32_v13 import V11Step2ProductionFP32V13


class V11Step2ProductionFP32V18(V11Step2ProductionFP32V13):
    """Retain four wall deadlines; frame pending depth remains exactly one."""

    def __init__(self) -> None:
        super().__init__()
        self.credit_capacity = 4
        print(
            "CAMERA_V11_STEP2_V18_POLICY credit_capacity=4 credit_type=deadline-not-frame "
            "gstreamer_pending_max=1 python_frame_queue=0",
            flush=True,
        )


def main() -> int:
    service = V11Step2ProductionFP32V18()

    def stop(_signum, _frame) -> None:
        service.stop_requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        return service.run()
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
