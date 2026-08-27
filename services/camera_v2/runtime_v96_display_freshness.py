from __future__ import annotations

from .runtime_v95_pts_audit import PascalPtsAuditRuntime


class PascalDisplayFreshnessRuntime(PascalPtsAuditRuntime):
    """V9.6: fix only the display branch freshness boundary.

    V9.5 proved that source->display P95 is hundreds of milliseconds even though
    each per-source display queue is latest-only.  The old graph placed the only
    post-mux leaky queue *after* nvmultistreamtiler:

        display_mux -> tiler -> latest_queue -> convert -> OSD -> EGL

    If tiler/downstream work back-pressures display_mux, that queue is too late to
    protect the mux from building stale batched output.  V9.6 moves the existing
    one-buffer downstream-leaky queue directly after display_mux:

        display_mux -> latest_queue -> tiler -> convert -> OSD -> EGL

    No queue limits, mux timing, tracker, detector, bbox, PTS instrumentation, or
    display dimensions change.  This is a one-behavior A/B test.
    """

    def __init__(self) -> None:
        super().__init__()
        print(
            "CAMERA_V96_ARCH only_change=display-queue-position "
            "old=display_mux->tiler->latest_queue "
            "new=display_mux->latest_queue->tiler "
            "queue=max1,leaky-downstream pts_audit=v95",
            flush=True,
        )

    def _link_display_path(self) -> None:
        # The existing wall_queue is already configured by _configure_display_path
        # as max-size-buffers=1, max-size-bytes/time=0, leaky=downstream.  Moving it
        # before the tiler creates the streaming-thread boundary at the point where
        # freshness is needed: immediately after display_mux.
        self._require_link(self.display_mux, self.wall_queue, "display mux -> latest queue")
        self._require_link(self.wall_queue, self.tiler, "latest queue -> tiler")
        self._require_link(self.tiler, self.wall_convert, "tiler -> display convert")
        self._require_link(self.wall_convert, self.wall_caps, "convert -> RGBA caps")
        self._require_link(self.wall_caps, self.osd, "RGBA -> OSD")
        self._require_link(self.osd, self.sink, "OSD -> EGL")

    def _audit_graph(self) -> None:
        display_peer = self.display_mux.get_static_pad("src").get_peer()
        tracker_peer = self.tracker_mux.get_static_pad("src").get_peer()
        queue_peer = self.wall_queue.get_static_pad("src").get_peer()

        display_name = (
            display_peer.get_parent_element().get_name() if display_peer is not None else None
        )
        tracker_name = (
            tracker_peer.get_parent_element().get_name() if tracker_peer is not None else None
        )
        queue_name = (
            queue_peer.get_parent_element().get_name() if queue_peer is not None else None
        )

        if display_name != self.wall_queue.get_name():
            raise RuntimeError(
                f"CAMERA_V96_AUDIT display mux peer={display_name}, "
                f"expected={self.wall_queue.get_name()}"
            )
        if queue_name != self.tiler.get_name():
            raise RuntimeError(
                f"CAMERA_V96_AUDIT display queue peer={queue_name}, "
                f"expected={self.tiler.get_name()}"
            )
        if tracker_name != self.tracker.get_name():
            raise RuntimeError(
                f"CAMERA_V96_AUDIT tracker mux peer={tracker_name}, "
                f"expected={self.tracker.get_name()}"
            )
        if self.pipeline.get_by_name("native_yolo26_pgie") is not None:
            raise RuntimeError("CAMERA_V96_AUDIT Gst-nvinfer must be absent on Pascal")

        for index, camera in enumerate(self.cameras):
            tee = self.pipeline.get_by_name(f"tee_{index}")
            detector_convert = self.pipeline.get_by_name(f"detector_convert_{index}")
            detector_sink = self.pipeline.get_by_name(f"detector_sink_{index}")
            if tee is None or detector_convert is None or detector_sink is None:
                raise RuntimeError(f"CAMERA_V96_AUDIT {camera.camera_id} branch missing")

        print(
            "CAMERA_V96_AUDIT status=OK "
            "display=display_mux->latest_queue->tiler->OSD->EGL "
            "tracker=tracker_mux->NvDCF->fakesink detector=inprocess-TRT86",
            flush=True,
        )


def main() -> int:
    return PascalDisplayFreshnessRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
