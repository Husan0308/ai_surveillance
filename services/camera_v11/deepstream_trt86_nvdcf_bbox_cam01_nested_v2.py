from __future__ import annotations

import os
from pathlib import Path

from services.camera_v11.deepstream_trt86_nvdcf_bbox_active_dedupe_v1 import (
    V11DeepStreamTRT86NvDCFActiveDedupeV1,
)


DEFAULT_CAM01_NESTED_CONFIG = str(
    Path(__file__).resolve().parents[2] / "config" / "camera_v11_bbox_nvdcf_cam01_nested_v2.yml"
)


class V11DeepStreamTRT86NvDCFBBoxCam01NestedV2(V11DeepStreamTRT86NvDCFActiveDedupeV1):
    """Apply CAM-01 nested-person tracking plus detector-guided active dedupe.

    CAM-01 keeps its separate NvDCF candidacy profile. The downstream active-dedupe
    layer removes a duplicate ACTIVE OSD box only when two strongly-overlapping
    tracks are not supported by two distinct recent detector-person boxes. This
    preserves two real nested people while enforcing one displayed box per person.
    CAM-02..06 keep their previously accepted behavior.
    """

    def __init__(self) -> None:
        self.cam01_nested_config = os.environ.get(
            "V11_BBOX_CAM01_NESTED_CONFIG", DEFAULT_CAM01_NESTED_CONFIG
        )
        super().__init__()
        print(
            "CAMERA_V11_BBOX_CAM01_NESTED_V2_ARCH "
            f"camera=CAM-01 config={self.cam01_nested_config} "
            "detector_floor=0.18 active_dedupe=detector-guided shared_profile_unchanged=1 "
            "tracker_feedback=normal",
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
