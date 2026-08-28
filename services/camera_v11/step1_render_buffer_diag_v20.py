from __future__ import annotations

import os

from .step1_render_path_diag_v20 import V11Step1RenderPathDiagV20


class V11Step1RenderBufferDiagV20(V11Step1RenderPathDiagV20):
    """CAM03-only RTSP jitter-buffer mode A/B; all other V7 settings frozen."""

    def __init__(self) -> None:
        self.render_buffer_mode = int(os.environ.get("V11_CAM03_BUFFER_MODE", "1"))
        if self.render_buffer_mode not in {1, 4}:
            raise RuntimeError("V11_CAM03_BUFFER_MODE must be 1 (slave) or 4 (synced)")
        super().__init__()
        name = "slave" if self.render_buffer_mode == 1 else "synced"
        print(
            f"CAMERA_V11_STEP1_RENDER_BUFFER_AB camera=CAM-03 mode={name} "
            "latency_ms=100 queue_max=1 interpolation=4",
            flush=True,
        )

    def _configure_rtsp_child(self, bin_obj, sub_bin, element, camera) -> None:
        super()._configure_rtsp_child(bin_obj, sub_bin, element, camera)
        factory = element.get_factory()
        if camera.camera_id != "CAM-03" or factory is None or factory.get_name() != "rtspsrc":
            return
        self._set_if(element, "buffer-mode", self.render_buffer_mode)
        effective = int(element.get_property("buffer-mode"))
        print(
            f"CAMERA_V11_STEP1_RENDER_BUFFER_EFFECTIVE camera=CAM-03 mode={effective}",
            flush=True,
        )


def main() -> int:
    return V11Step1RenderBufferDiagV20().run()


if __name__ == "__main__":
    raise SystemExit(main())
