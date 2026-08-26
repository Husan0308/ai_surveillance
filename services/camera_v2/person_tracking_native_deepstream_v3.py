from __future__ import annotations

from pathlib import Path

from . import person_tracking_native_deepstream as native
from .person_tracking_native_deepstream_v2 import _native_tracker_config_v2


class CameraPersonTrackingNativeDeepStreamV3(native.CameraPersonTrackingNativeDeepStream):
    """Native DeepStream graph with verified object/pad rewiring.

    GStreamer graph mutations are verified through the actual object hierarchy and
    pad peers instead of relying on Python binding return-value quirks.  Elements
    must be parented by the top-level pipeline before any link is attempted.
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

    def _parent_text(self, element) -> str:
        parent = element.get_parent()
        if parent is None:
            return "none"
        try:
            return parent.get_name()
        except Exception:
            return str(parent)

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
        # GStreamer requires both elements to belong to the same bin before
        # gst_element_link(). Verify that invariant explicitly.
        src_parent = src.get_parent()
        dst_parent = dst.get_parent()
        if src_parent is None or dst_parent is None:
            raise RuntimeError(
                f"{label}: element not parented before link "
                f"src_parent={self._parent_text(src)} dst_parent={self._parent_text(dst)}"
            )
        if src_parent.get_name() != self.pipeline.get_name() or dst_parent.get_name() != self.pipeline.get_name():
            raise RuntimeError(
                f"{label}: elements are not children of the top-level pipeline "
                f"src_parent={self._parent_text(src)} dst_parent={self._parent_text(dst)}"
            )

        if not src.link(dst):
            raise RuntimeError(
                f"{label}: Gst.Element.link failed "
                f"src={src.get_name()} dst={dst.get_name()} "
                f"src_peer={self._peer_text(self._static_pad(src, 'src', label))} "
                f"dst_peer={self._peer_text(self._static_pad(dst, 'sink', label))}"
            )
        self._verify_direct_link(src, dst, label)

    def _add_required(self, element, label: str) -> None:
        name = element.get_name()
        before_parent = element.get_parent()
        if before_parent is not None:
            raise RuntimeError(
                f"{label}: {name} already has parent={self._parent_text(element)} before add"
            )

        existing = self.pipeline.get_by_name(name)
        if existing is not None:
            raise RuntimeError(
                f"{label}: pipeline already contains another element named {name}"
            )

        # gst_bin_add() is documented to parent the element to the bin.  Some GI
        # environments have produced misleading return values in this deployment,
        # so the object hierarchy after the call is the authoritative check.
        add_result = self.pipeline.add(element)
        after_parent = element.get_parent()
        if after_parent is None:
            raise RuntimeError(
                f"{label}: add failed for {name}; result={add_result!r} parent=none"
            )
        if after_parent.get_name() != self.pipeline.get_name():
            raise RuntimeError(
                f"{label}: {name} attached to unexpected parent={self._parent_text(element)} "
                f"result={add_result!r}"
            )

        resolved = self.pipeline.get_by_name(name)
        if resolved is None or resolved.get_name() != name:
            raise RuntimeError(
                f"{label}: {name} parented but not resolvable from pipeline"
            )

        print(
            f"CAMERA_NATIVE_ELEMENT_ADD label={label.replace(' ', '_')} "
            f"name={name} result={add_result!r} parent={after_parent.get_name()} verified=1",
            flush=True,
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
    # Use the section-aware DS7.1 NvDCF generator and the object/pad-verified graph.
    native._native_tracker_config = _native_tracker_config_v2
    return CameraPersonTrackingNativeDeepStreamV3().run()


if __name__ == "__main__":
    raise SystemExit(main())
