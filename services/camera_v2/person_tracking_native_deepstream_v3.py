from __future__ import annotations

from pathlib import Path

from . import person_tracking_native_deepstream as native
from .person_tracking_native_deepstream_v2 import _native_tracker_config_v2


class CameraPersonTrackingNativeDeepStreamV3(native.CameraPersonTrackingNativeDeepStream):
    """Native DeepStream graph with verified static-pad rewiring.

    Gst.Element.unlink() has no boolean return value.  The previous runtime treated
    its Python return value as success/failure, so every valid unlink looked like a
    failure.  This version verifies the actual pad peers before and after rewiring.
    """

    def _static_pad(self, element, name: str, label: str):
        pad = element.get_static_pad(name)
        if pad is None:
            raise RuntimeError(
                f"{label}: missing static pad {element.get_name()}.{name}"
            )
        return pad

    def _peer_text(self, pad) -> str:
        peer = pad.get_peer()
        if peer is None:
            return "none"
        parent = peer.get_parent_element()
        parent_name = parent.get_name() if parent is not None else "?"
        return f"{parent_name}.{peer.get_name()}"

    def _verify_direct_link(self, src, dst, label: str) -> None:
        src_pad = self._static_pad(src, "src", label)
        dst_pad = self._static_pad(dst, "sink", label)
        src_peer = src_pad.get_peer()
        dst_peer = dst_pad.get_peer()
        if src_peer is None or dst_peer is None:
            raise RuntimeError(
                f"{label}: expected direct link but peer is missing "
                f"src_peer={self._peer_text(src_pad)} dst_peer={self._peer_text(dst_pad)}"
            )
        src_parent = src_peer.get_parent_element()
        dst_parent = dst_peer.get_parent_element()
        if (
            src_parent is None
            or dst_parent is None
            or src_parent.get_name() != dst.get_name()
            or dst_parent.get_name() != src.get_name()
        ):
            raise RuntimeError(
                f"{label}: unexpected peer mapping "
                f"src_peer={self._peer_text(src_pad)} dst_peer={self._peer_text(dst_pad)}"
            )

    def _unlink_required(self, src, dst, label: str) -> None:
        self._verify_direct_link(src, dst, label)
        # Gst.Element.unlink() is void in GStreamer; do not test its return value.
        src.unlink(dst)
        src_pad = self._static_pad(src, "src", label)
        dst_pad = self._static_pad(dst, "sink", label)
        if src_pad.is_linked() or dst_pad.is_linked():
            raise RuntimeError(
                f"{label}: unlink did not detach pads "
                f"src_peer={self._peer_text(src_pad)} dst_peer={self._peer_text(dst_pad)}"
            )

    def _link_required(self, src, dst, label: str) -> None:
        if not src.link(dst):
            raise RuntimeError(
                f"{label}: Gst.Element.link failed "
                f"src={src.get_name()} dst={dst.get_name()}"
            )
        self._verify_direct_link(src, dst, label)

    def _add_required(self, element, label: str) -> None:
        if not self.pipeline.add(element):
            raise RuntimeError(
                f"{label}: could not add {element.get_name()} to pipeline"
            )

    def _install_native_analytics(
        self, pgie_config: Path, tracker_lib: Path, tracker_config: Path
    ) -> None:
        # Base graph created by DynamicCameraWallV2 is:
        # mux -> tiler -> wall_caps -> wall_queue -> sink.
        # Rewire it while the pipeline is still NULL, before run() changes state.
        self._unlink_required(
            self.mux, self.tiler, "native rewire nvstreammux -> tiler"
        )
        self._unlink_required(
            self.wall_queue, self.sink, "native rewire wall_queue -> sink"
        )

        mux_queue = self._make("queue", "native_mux_pgie_queue")
        pgie = self._make("nvinfer", "native_yolo26_pgie")
        track_queue = self._make("queue", "native_pgie_tracker_queue")
        tracker = self._make("nvtracker", "native_nvdcf_tracker")
        convert = self._make("nvvideoconvert", "native_wall_convert")
        rgba_caps = self._make("capsfilter", "native_wall_rgba")
        osd = self._make("nvdsosd", "native_wall_osd")

        self._configure_queue(mux_queue)
        self._configure_queue(track_queue)

        pgie.set_property("config-file-path", str(pgie_config))
        self._set_if(pgie, "batch-size", len(self.cameras))
        self._set_if(pgie, "interval", 19)
        self._set_if(pgie, "gpu-id", self.gpu_id)

        self._set_if(tracker, "tracker-width", 640)
        self._set_if(tracker, "tracker-height", 384)
        tracker.set_property("ll-lib-file", str(tracker_lib))
        tracker.set_property("ll-config-file", str(tracker_config))
        self._set_if(tracker, "gpu-id", self.gpu_id)
        self._set_if(tracker, "compute-hw", 1)
        self._set_if(tracker, "enable-batch-process", True)
        self._set_if(tracker, "display-tracking-id", False)
        self._set_if(tracker, "tracking-id-reset-mode", 1)

        self._set_if(convert, "gpu-id", self.gpu_id)
        rgba_caps.set_property(
            "caps", self.Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
        )
        self._set_if(osd, "process-mode", 1)
        self._set_if(osd, "display-bbox", True)
        self._set_if(osd, "display-text", False)
        self._set_if(osd, "display-mask", False)
        self._set_if(osd, "gpu-id", self.gpu_id)

        for element, label in (
            (mux_queue, "mux queue"),
            (pgie, "primary nvinfer"),
            (track_queue, "tracker queue"),
            (tracker, "NvDCF tracker"),
            (convert, "wall converter"),
            (rgba_caps, "wall RGBA caps"),
            (osd, "wall OSD"),
        ):
            self._add_required(element, label)

        analytics_chain = [
            self.mux,
            mux_queue,
            pgie,
            track_queue,
            tracker,
            self.tiler,
        ]
        for src, dst in zip(analytics_chain, analytics_chain[1:]):
            self._link_required(
                src,
                dst,
                f"native analytics {src.get_name()} -> {dst.get_name()}",
            )

        # The base tiler -> wall_caps -> wall_queue links were never disturbed.
        self._verify_direct_link(
            self.tiler, self.wall_caps, "native display tiler -> wall_caps"
        )
        self._verify_direct_link(
            self.wall_caps, self.wall_queue, "native display wall_caps -> wall_queue"
        )

        display_chain = [self.wall_queue, convert, rgba_caps, osd, self.sink]
        for src, dst in zip(display_chain, display_chain[1:]):
            self._link_required(
                src,
                dst,
                f"native display {src.get_name()} -> {dst.get_name()}",
            )

        tracker_src = self._static_pad(tracker, "src", "native tracker probe")
        tracker_src.add_probe(self.Gst.PadProbeType.BUFFER, self._native_tracker_probe)

        self.pgie = pgie
        self.tracker = tracker
        self.osd = osd

        print(
            "CAMERA_NATIVE_GRAPH verified=1 "
            "chain=nvstreammux->queue->nvinfer->queue->nvtracker->tiler->"
            "wall_caps->wall_queue->nvvideoconvert->RGBA->nvdsosd->sink",
            flush=True,
        )


def main() -> int:
    # Use the section-aware DS7.1 NvDCF generator and the pad-verified graph.
    native._native_tracker_config = _native_tracker_config_v2
    return CameraPersonTrackingNativeDeepStreamV3().run()


if __name__ == "__main__":
    raise SystemExit(main())
