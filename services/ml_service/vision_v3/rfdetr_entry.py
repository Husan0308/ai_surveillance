from __future__ import annotations

import sys

from . import rfdetr_detection as runtime
from .stable_boxes import StableFullBodyManager

# Keep the runtime module focused on detector logic while this entrypoint owns
# process-level stderr/exit handling.
runtime.sys = sys


def _install_osd_pygi(self) -> None:
    """Insert nvdsosd into the already-built camera display chain.

    Gst.Element.unlink() maps the C API gst_element_unlink(), whose return type is
    void. PyGObject therefore returns None even when the unlink succeeds. The old
    `if not element.unlink(...)` check treated every successful unlink as failure
    and aborted before the pipeline could start.
    """
    wall_src = self.wall_queue.get_static_pad("src")
    sink_pad = self.sink.get_static_pad("sink")
    if wall_src is None or sink_pad is None:
        raise RuntimeError("camera-core display pads are unavailable")

    peer = wall_src.get_peer()
    if peer is not None:
        if peer != sink_pad:
            raise RuntimeError("camera-core wall queue is linked to an unexpected element")
        wall_src.unlink(peer)

    if wall_src.is_linked() or sink_pad.is_linked():
        raise RuntimeError("could not detach camera-core sink for RF-DETR OSD")

    convert = self._make("nvvideoconvert", "v3_detect_wall_convert")
    caps = self._make("capsfilter", "v3_detect_wall_caps")
    osd = self._make("nvdsosd", "v3_detect_osd")

    self._set_if(convert, "gpu-id", self.gpu_id)
    caps.set_property("caps", self.Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))
    self._set_if(osd, "process-mode", 1)
    self._set_if(osd, "display-bbox", True)
    self._set_if(osd, "display-text", False)
    self._set_if(osd, "display-mask", False)
    self._set_if(osd, "gpu-id", self.gpu_id)

    for element in (convert, caps, osd):
        self.pipeline.add(element)

    if not self.wall_queue.link(convert):
        raise RuntimeError("failed wall queue -> nvvideoconvert")
    if not convert.link(caps):
        raise RuntimeError("failed nvvideoconvert -> RGBA caps")
    if not caps.link(osd):
        raise RuntimeError("failed RGBA caps -> nvdsosd")
    if not osd.link(self.sink):
        raise RuntimeError("failed nvdsosd -> nveglglessink")

    mux_src = self.mux.get_static_pad("src")
    osd_src = osd.get_static_pad("src")
    if mux_src is None or osd_src is None:
        raise RuntimeError("RF-DETR OSD probe pads are unavailable")
    mux_src.add_probe(self.Gst.PadProbeType.BUFFER, self._inject_boxes_probe)
    osd_src.add_probe(self.Gst.PadProbeType.BUFFER, self._wall_probe)
    self.osd = osd


# Apply compatibility + the proven core-v1 visual-continuity design before
# SixCameraRFDETR is constructed. Raw RF-DETR boxes are still untouched; only
# the display manager is replaced.
runtime.SixCameraRFDETR._install_osd = _install_osd_pygi
runtime.ProtectiveBoxManager = StableFullBodyManager


def main() -> int:
    return runtime.main()


if __name__ == "__main__":
    raise SystemExit(main())
