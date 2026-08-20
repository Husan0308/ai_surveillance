from __future__ import annotations

"""Pascal-safe Camera V2 runtime adapter.

DeepStream 7.1's validated dGPU matrix does not include Pascal/GTX 10-series.
On the deployment GTX 1050 Ti the RTSP/NVDEC/mux path is healthy, while NvDCF
accepts mux input and then stops producing downstream buffers.  This adapter
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
        # Use the known-good detector presentation chain directly.  This inserts
        # no nvtracker element, so unsupported NvDCF CUDA/pixel processing cannot
        # block mux output before the tiler/sink.
        CameraDetectionV2._install_osd_and_meta(self)
        self.tracker = None
        self.tracker_backend = "motion-predictor"
        print(
            "CAMERA_TRACK_FALLBACK backend=motion-predictor nvtracker=disabled "
            "reason=pascal-deepstream71-safe-mode",
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
