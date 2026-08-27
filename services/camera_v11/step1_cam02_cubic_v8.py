from __future__ import annotations

import os

from .step1_independent_egl_v4 import V11Step1IndependentEglV4


class V11Step1Cam02CubicV8(V11Step1IndependentEglV4):
    """V4 independent EGL display with one controlled scaler change.

    Baseline stays TCP/100 ms, six independent pipelines, latest-only queue,
    NVDEC, NVMM and nveglglessink. All cameras keep GPU Lanczos except CAM-02,
    whose 3200x1800 source is downscaled with GPU Cubic to test whether the
    expensive Lanczos transform is the reason decoded 20 FPS becomes ~16 FPS
    at render.
    """

    def __init__(self) -> None:
        self.cam02_interpolation = max(
            0, min(6, int(os.environ.get("V11_CAM02_INTERPOLATION", "2")))
        )
        super().__init__()

        cid = "CAM-02"
        convert = self.converters.get(cid)
        if convert is None:
            raise RuntimeError("V11 Step1 V8 CAM-02 converter missing")
        prop = convert.find_property("interpolation-method")
        if prop is None:
            raise RuntimeError("V11 Step1 V8 nvvideoconvert interpolation-method missing")
        convert.set_property("interpolation-method", self.cam02_interpolation)
        effective = int(convert.get_property("interpolation-method"))
        if effective != self.cam02_interpolation:
            raise RuntimeError(
                f"V11 Step1 V8 CAM-02 interpolation readback mismatch "
                f"expected={self.cam02_interpolation} got={effective}"
            )

        print(
            "CAMERA_V11_STEP1V8_ARCH base=v4-independent-egl scaler_ab=1 "
            "mux=0 tiler=0 detector=0 tracker=0 latest_only=1 transport=tcp",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP1V8_POLICY latency_ms=100 "
            f"others_interpolation={self.interpolation} cam02_interpolation={effective} "
            "cam02_source=3200x1800 target=640x360",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP1V8_SCALER camera=CAM-02 "
            f"interpolation={effective} expected={self.cam02_interpolation} "
            "gpu_scaling=1 single_resize=1",
            flush=True,
        )


def main() -> int:
    return V11Step1Cam02CubicV8().run()


if __name__ == "__main__":
    raise SystemExit(main())
