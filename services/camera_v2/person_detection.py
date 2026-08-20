from __future__ import annotations

from .detection import CameraDetectionV2


class CameraPersonDetectionV2(CameraDetectionV2):
    """Stable entrypoint for Camera V2 + YOLO26m person detection.

    This keeps the known-good camera wall intact through the tiler/latest-only
    queue, then adds one final GPU color conversion + nvdsosd stage for boxes.
    """

    def _install_osd_and_meta(self) -> None:
        # Gst.Element.unlink() is void in PyGObject; call it directly.
        self.unlink_display_source(self.wall_queue)

        convert = self._make("nvvideoconvert", "detect_wall_convert")
        caps = self._make("capsfilter", "detect_wall_caps")
        osd = self._make("nvdsosd", "detect_osd")

        self._set_if(convert, "gpu-id", self.gpu_id)
        caps.set_property(
            "caps",
            self.Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
        )
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
        if not self.link_display_source(osd):
            raise RuntimeError("failed nvdsosd -> display adapter")

        self.mux.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._inject_boxes_probe,
        )
        osd.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._wall_probe,
        )
        self.osd = osd


def main() -> int:
    return CameraPersonDetectionV2().run()


if __name__ == "__main__":
    raise SystemExit(main())
