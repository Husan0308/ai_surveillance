from __future__ import annotations

"""Pascal-safe Camera V2 runtime adapter.

DeepStream 7.1's validated dGPU matrix does not include Pascal/GTX 10-series.
On the deployment GTX 1050 Ti the RTSP/NVDEC/mux path is healthy, while NvDCF
accepts mux input and then stops producing downstream buffers. This adapter
therefore removes gst-nvtracker from the hot path on that machine and reuses the
existing CameraDetectionV2 motion stabilizer between sparse RF-DETR observations.

The video path remains GPU-native:
RTSP -> NVDEC -> nvstreammux -> nvmultistreamtiler -> NVMM OSD -> EGL.
Only the unsupported NvDCF stage is bypassed.
"""

import os
import time


def _enabled() -> bool:
    return os.environ.get("CAMERA_V2_PASCAL_SAFE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def install_pascal_safe_pipeline() -> bool:
    """Install the no-NvDCF fallback before CameraPersonTrackingFinal is built."""
    if not _enabled():
        return False

    from .detection import CameraDetectionV2
    from .person_tracking import CameraPersonTrackingV2
    from .person_tracking_final import CameraPersonTrackingFinal

    if getattr(CameraPersonTrackingFinal, "_pascal_safe_installed", False):
        return True

    def _install_osd_without_nvtracker(self) -> None:
        """Install the proven GPU OSD chain without calling the old unlink bug.

        PyGObject's Gst.Element.unlink() returns None even when unlink succeeds.
        The old CameraDetectionV2 implementation treated that return value as a
        boolean and raised `could not detach baseline sink for OSD` on a successful
        unlink. Verify the actual pad peer instead.
        """
        wall_src = self.wall_queue.get_static_pad("src")
        sink_pad = self.sink.get_static_pad("sink")
        if wall_src is None or sink_pad is None:
            raise RuntimeError("Pascal-safe OSD: baseline display pads unavailable")

        peer = wall_src.get_peer()
        if peer is not None:
            if peer != sink_pad:
                peer_name = peer.get_parent_element().get_name() if peer.get_parent_element() else "unknown"
                raise RuntimeError(
                    f"Pascal-safe OSD: wall queue linked to unexpected peer {peer_name}"
                )
            wall_src.unlink(peer)

        if wall_src.is_linked() or sink_pad.is_linked():
            raise RuntimeError("Pascal-safe OSD: baseline sink did not detach")

        convert = self._make("nvvideoconvert", "pascal_detect_wall_convert")
        caps = self._make("capsfilter", "pascal_detect_wall_caps")
        osd = self._make("nvdsosd", "pascal_detect_osd")

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
            raise RuntimeError("Pascal-safe OSD: wall queue -> nvvideoconvert failed")
        if not convert.link(caps):
            raise RuntimeError("Pascal-safe OSD: nvvideoconvert -> RGBA caps failed")
        if not caps.link(osd):
            raise RuntimeError("Pascal-safe OSD: RGBA caps -> nvdsosd failed")
        if not osd.link(self.sink):
            raise RuntimeError("Pascal-safe OSD: nvdsosd -> EGL sink failed")

        mux_src = self.mux.get_static_pad("src")
        osd_src = osd.get_static_pad("src")
        if mux_src is None or osd_src is None:
            raise RuntimeError("Pascal-safe OSD: probe pads unavailable")

        # Dynamic dispatch is intentional. _inject_boxes_probe is replaced below
        # with the counter-aware Pascal-safe variant before runtime construction.
        mux_src.add_probe(self.Gst.PadProbeType.BUFFER, self._inject_boxes_probe)
        osd_src.add_probe(self.Gst.PadProbeType.BUFFER, self._wall_probe)

        self.tracker = None
        self.tracker_backend = "motion-predictor"
        self.osd = osd
        print(
            "CAMERA_TRACK_FALLBACK backend=motion-predictor nvtracker=disabled "
            "reason=pascal-deepstream71-safe-mode osd_link=safe-pad-peer",
            flush=True,
        )

    def _active_motion_counts(self) -> dict[int, int]:
        now = time.monotonic()
        output: dict[int, int] = {
            int(source_id): 0 for source_id in self.camera_index.values()
        }
        boxes = getattr(self, "boxes", None)
        if boxes is None:
            return output
        with boxes.lock:
            for cid, source_id in self.camera_index.items():
                active = 0
                for track in boxes.tracks.get(cid, {}).values():
                    if now - float(track.last_det_t) <= float(boxes.max_age):
                        active += 1
                output[int(source_id)] = active
        return output

    def _inject_boxes_with_counts(self, pad, info):
        result = CameraDetectionV2._inject_boxes_probe(self, pad, info)
        counts = _active_motion_counts(self)
        with self.det_lock:
            self.source_track_counts = counts
            self.tracked_now = sum(counts.values())
            self.tracker_frames += 1
        return result

    def _pascal_print_stats(self) -> bool:
        keep = CameraDetectionV2._print_stats(self)
        counts = _active_motion_counts(self)
        with self.det_lock:
            self.source_track_counts = counts
            self.tracked_now = sum(counts.values())
            tracked_now = int(self.tracked_now)
            frames = int(self.tracker_frames)
        print(
            "CAMERA_TRACK_FALLBACK "
            f"backend=motion-predictor tracked_now={tracked_now} "
            f"wall_batches={frames} source_counts={counts} nvtracker=0",
            flush=True,
        )
        return keep

    def _pascal_live_source_counts(self) -> dict[int, int]:
        counts = _active_motion_counts(self)
        with self.det_lock:
            self.source_track_counts = counts
            self.tracked_now = sum(counts.values())
        return counts

    # CameraDetectionV2.__init__ dispatches _install_osd_and_meta dynamically.
    # Replacing the tracking-class implementation before construction prevents
    # gst-nvtracker from ever being inserted into this process.
    CameraPersonTrackingV2._install_osd_and_meta = _install_osd_without_nvtracker

    # The Final scheduler normally publishes detections into NvDCF metadata.
    # In safe mode use the already-tested detector scheduler, which updates the
    # per-camera SmoothBoxManager and lets its prediction render on every mux frame.
    CameraPersonTrackingFinal._scheduler = CameraDetectionV2._scheduler
    CameraPersonTrackingFinal._inject_boxes_probe = _inject_boxes_with_counts
    CameraPersonTrackingFinal._print_stats = _pascal_print_stats
    CameraPersonTrackingFinal.live_source_counts = _pascal_live_source_counts
    CameraPersonTrackingFinal._pascal_safe_installed = True
    return True
