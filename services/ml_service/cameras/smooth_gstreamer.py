from __future__ import annotations

import threading

from . import gstreamer as base


_BUILD_LOCK = threading.Lock()


def _inject_gpu_scale(pipeline: str, config: dict) -> str:
    width = max(0, int(config.get("capture_output_width", 0) or 0))
    height = max(0, int(config.get("capture_output_height", 0) or 0))
    if width <= 0 or height <= 0:
        return pipeline

    # The nvv4l2decoder path already reaches nvvideoconvert in NVMM. Constrain
    # its RAW output here so NVIDIA performs scale/color conversion before the
    # Python appsink mapping. If nvcodec is selected, leave the existing path
    # untouched rather than forcing an unsupported element combination.
    needle = "nvvideoconvert name=converter ! video/x-raw,format=BGRx"
    replacement = (
        "nvvideoconvert name=converter ! "
        f"video/x-raw,width={width},height={height},format=BGRx"
    )
    pipeline = pipeline.replace(needle, replacement)
    pipeline = pipeline.replace(
        "appsink name=sink drop=true max-buffers=1 sync=false wait-on-eos=false",
        "appsink name=sink drop=true max-buffers=1 sync=false wait-on-eos=false enable-last-sample=false",
    )
    return pipeline


class SmoothGStreamerCapture(base.GStreamerCapture):
    backend = "gstreamer-nvdec-smooth"

    def __init__(self, config: dict):
        # GStreamerCapture resolves nvidia_rtsp_pipeline from module globals.
        # Serialize construction while temporarily wrapping that builder so all
        # six camera threads get deterministic pipelines without a global race.
        with _BUILD_LOCK:
            original = base.nvidia_rtsp_pipeline
            try:
                base.nvidia_rtsp_pipeline = lambda cfg: _inject_gpu_scale(original(cfg), cfg)
                super().__init__(config)
            finally:
                base.nvidia_rtsp_pipeline = original
        self._source_runtime.update(
            {
                "backend": self.backend,
                "capture_output_width": int(config.get("capture_output_width", 0) or 0),
                "capture_output_height": int(config.get("capture_output_height", 0) or 0),
                "gpu_scale_before_host_copy": "nvvideoconvert" in self.pipeline
                and "width=" in self.pipeline
                and "height=" in self.pipeline,
            }
        )
