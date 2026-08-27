from __future__ import annotations

import os

from .step2_detector_b1_v5 import V11Step2DetectorB1V5


class V11Step2DetectorB1V6(V11Step2DetectorB1V5):
    """Step2 V6: preserve V5 topology and reduce detector duty to 2 Hz/camera.

    This is an A/B isolation step. No display, RTSP, decoder, queue, conversion,
    TensorRT engine, or detector geometry is changed. Only the batch-1 round-robin
    request rate is reduced so we can verify whether GPU duty caused the Step1 V7
    display regression seen at 3 Hz/camera.
    """

    def __init__(self) -> None:
        # Fixed for this A/B branch. Do not let shell/user state silently change it.
        os.environ["V11_DETECT_B1_HZ_PER_CAMERA"] = "2.0"
        super().__init__()
        if abs(self.b1_target_hz_per_camera - 2.0) > 1e-6:
            raise RuntimeError(
                f"V11 Step2 V6 must run exactly 2.0 Hz/camera, got {self.b1_target_hz_per_camera}"
            )
        print(
            "CAMERA_V11_STEP2V6_ARCH "
            "base=step2-v5-topology display=step1-v7-frozen detector=trt86-batch1 "
            "scheduler=round-robin-jit batch=1 prefetch=0 tracker=0 osd=0 reid=0 face=0",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP2V6_POLICY "
            "per_camera_target=2.00Hz global_target=12.00Hz change=detector-duty-only "
            "display_changed=0 conversion_changed=0 engine_changed=0",
            flush=True,
        )


def main() -> int:
    return V11Step2DetectorB1V6().run()


if __name__ == "__main__":
    raise SystemExit(main())
