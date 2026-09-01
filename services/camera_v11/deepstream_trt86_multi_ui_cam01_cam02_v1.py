from __future__ import annotations

from services.camera_v11.deepstream_trt86_multi_ui_v1 import (
    DEFAULT_UI_CAMERAS,
    V11DeepStreamTRT86MultiCameraUIV1,
    _camera_env_key,
    _default_preview_path,
)


class V11DeepStreamTRT86MultiCameraUICam01Cam02V1(V11DeepStreamTRT86MultiCameraUIV1):
    """Compatibility entry point for the proven CAM-01/CAM-02 preview milestone."""


def main() -> int:
    return V11DeepStreamTRT86MultiCameraUICam01Cam02V1().run()


if __name__ == "__main__":
    raise SystemExit(main())
