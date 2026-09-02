from __future__ import annotations

import os
from pathlib import Path

from services.camera_v11.deepstream_trt86_nvdcf_bbox_shadow_display_v1 import (
    V11DeepStreamTRT86NvDCFShadowDisplayV1,
)


DEFAULT_CAM01_NESTED_CONFIG = str(
    Path(__file__).resolve().parents[2] / "config" / "camera_v11_bbox_nvdcf_cam01_nested_v2.yml"
)


class V11DeepStreamTRT86NvDCFBBoxCam01NestedV2(V11DeepStreamTRT86NvDCFShadowDisplayV1):
    """Apply a CAM-01-only NvDCF candidacy profile without touching CAM-02..06.

    The six-camera detector, shadow-display behavior, association weights, HOG,
    ReAssoc, tracker resolution, and all other accepted settings remain unchanged.
    Only CAM-01 gets a separate ll-config-file whose detector confidence floor is
    aligned with the external detector's 0.18 admission threshold.
    """

    def __init__(self) -> None:
        self.cam01_nested_config = os.environ.get(
            "V11_BBOX_CAM01_NESTED_CONFIG", DEFAULT_CAM01_NESTED_CONFIG
        )
        super().__init__()
        print(
            "CAMERA_V11_BBOX_CAM01_NESTED_V2_ARCH "
            f"camera=CAM-01 config={self.cam01_nested_config} "
            "detector_floor=0.18 shared_profile_unchanged=1 tracker_feedback=normal",
            flush=True,
        )

    def _preflight(self) -> None:
        super()._preflight()
        if not Path(self.cam01_nested_config).is_file():
            raise RuntimeError(
                f"CAM-01 nested NvDCF config missing: {self.cam01_nested_config}"
            )

    def _build_camera(self, state) -> None:
        cid = state.camera.camera_id
        if cid != "CAM-01":
            return super()._build_camera(state)

        # The parent tracker builder reads self.tracker_config when it creates the
        # nvtracker element. Swap it only for CAM-01, then immediately restore the
        # shared profile before the next camera is built. Camera pipelines are built
        # serially during startup, so no runtime tracker state is affected by this.
        shared_config = self.tracker_config
        self.tracker_config = self.cam01_nested_config
        try:
            return super()._build_camera(state)
        finally:
            self.tracker_config = shared_config


def main() -> int:
    return V11DeepStreamTRT86NvDCFBBoxCam01NestedV2().run()


if __name__ == "__main__":
    raise SystemExit(main())
