from __future__ import annotations

import os
import sys
import time

# Detection-only GTX 1050 Ti profile.
# The model stays YOLO26m and the detector stays strict batch=6, but reducing the
# per-batch pixel count shortens each CUDA burst. This matters more for visible
# camera stutter than merely waiting longer between two 200 ms bursts.
os.environ.setdefault("AI_YOLO_INFER_WIDTH", "576")
os.environ.setdefault("AI_YOLO_INFER_HEIGHT", "320")
os.environ.setdefault("AI_YOLO_PREDICT_WIDTH", "576")
os.environ.setdefault("AI_YOLO_PREDICT_HEIGHT", "320")
os.environ.setdefault("AI_YOLO_START_BATCH_FPS", "0.80")
os.environ.setdefault("AI_YOLO_MAX_BATCH_FPS", "1.05")
os.environ.setdefault("AI_YOLO_MIN_BATCH_FPS", "0.45")
os.environ.setdefault("AI_YOLO_MAX_GPU_DUTY", "0.18")
os.environ.setdefault("AI_YOLO_CONF", "0.18")
os.environ.setdefault("AI_CAMERA_FPS_FLOOR", "19.3")
os.environ.setdefault("AI_CAMERA_FPS_GOOD", "19.8")

# Live RTSP rendering must not use the EGL sink clock as a second frame-drop
# authority. nvstreammux already regulates the six inputs; the final sink should
# present the newest tiled frame immediately instead of declaring it "too late".
os.environ.setdefault("AI_WALL_SINK_SYNC", "0")

from . import deepstream_yolo26m_batch6_wall as base
from .dsmeta_bridge import DeepStreamMetaBridge


class NativeCameraYolo26mDetectionOnly(base.NativeCameraYolo26mBatch6Wall):
    """Only six-camera display + YOLO26m person detection + visible rectangles.

    Display:
        RTSP -> NVDEC -> tee -> latest-only queue -> nvstreammux (20 FPS input sync)
             -> tiler 1920x720 -> latest-only OSD queue -> nvdsosd GPU
             -> latest-only wall queue -> EGL (sync=0, no late-frame drops)

    Detector:
        tee -> ticket gate -> one fresh frame/camera -> 576x320 BGRx
            -> exactly six frames -> one YOLO26m CUDA call

    Bounding boxes are inserted with a tiny native C DeepStream metadata bridge,
    not pyds. This avoids the DeepStream-7.1/Python binding compatibility issue.

    No tracker, ReID, face, pose, heatmap, API, mmap, JPEG or Qt UI.
    """

    def __init__(self):
        self.meta_bridge: DeepStreamMetaBridge | None = None
        self.meta_bridge_error = ""
        self.osd = None
        self.osd_queue = None
        self.osd_enabled = False

        self.meta_max_age_ms = max(
            700.0,
            float(os.environ.get("AI_DETECTION_BOX_HOLD_MS", "3000")),
        )
        self.display_wall_width = max(
            960, int(os.environ.get("AI_DETECTION_WALL_WIDTH", "1920"))
        )
        self.display_wall_height = max(
            360, int(os.environ.get("AI_DETECTION_WALL_HEIGHT", "720"))
        )

        self._boxes_meta = 0
        self._bridge_frames = 0
        self._tiled_objects_last = 0
        self._tiled_objects_total = 0
        self._diag_next = 0.0

        super().__init__()

        # The six measured RTSP streams are ~20 FPS with ~50 ms PTS spacing.
        # Synchronize them at the mux, where we still have per-source context,
        # rather than at the final renderer where late PTS caused frame drops.
        self._set_if(self.mux, "batched-push-timeout", 50000)
        self._set_if(self.mux, "sync-inputs", True)
        self._set_if(self.mux, "max-latency", 120_000_000)  # 120 ms in ns
        self._set_if(self.mux, "buffer-pool-size", 12)
        self.mux_timeout_us = 50000

        # Do not render a 3840x1440 offscreen wall only to downscale it again in
        # the desktop/AnyDesk. 1920x720 is 75% fewer output pixels. The source
        # mux remains 1280x720, so source detail before tiling is unchanged.
        self._set_if(self.tiler, "width", self.display_wall_width)
        self._set_if(self.tiler, "height", self.display_wall_height)
        self.wall_width = self.display_wall_width
        self.wall_height = self.display_wall_height

        # DeepStream's RTSP troubleshooting guidance recommends sync=0 on display
        # sinks. GstBaseSink also disables clock-lateness dropping when sync is
        # false. Keep QoS off and explicitly remove every additional sink timing
        # constraint that could turn a small GPU/desktop hiccup into visible loss.
        self._set_if(self.sink, "sync", False)
        self._set_if(self.sink, "qos", False)
        self._set_if(self.sink, "max-lateness", -1)
        self._set_if(self.sink, "processing-deadline", 0)
        self._set_if(self.sink, "render-delay", 0)
        self._set_if(self.sink, "throttle-time", 0)
        self._set_if(self.sink, "enable-last-sample", False)
        self.sink_sync = False

        self._setup_native_bbox_osd()

    def _setup_native_bbox_osd(self) -> None:
        try:
            self.meta_bridge = DeepStreamMetaBridge()
            print(
                "DETECTION_ONLY native-meta=1 "
                f"bridge={self.meta_bridge.library_path} ds={self.meta_bridge.ds_root}",
                flush=True,
            )
        except Exception as exc:
            self.meta_bridge = None
            self.meta_bridge_error = f"{type(exc).__name__}: {exc}"
            print(
                "DETECTION_ONLY native-meta=0; YOLO works but boxes cannot be attached: "
                f"{self.meta_bridge_error}",
                file=sys.stderr,
                flush=True,
            )
            return

        osd_queue = self.Gst.ElementFactory.make("queue", "detection_osd_queue")
        osd = self.Gst.ElementFactory.make("nvdsosd", "detection_osd")
        if osd_queue is None or osd is None:
            self.meta_bridge_error = "nvdsosd GStreamer element is unavailable"
            print(
                "DETECTION_ONLY osd=0; native metadata works but nvdsosd plugin is unavailable",
                file=sys.stderr,
                flush=True,
            )
            return

        self._set_if(osd_queue, "max-size-buffers", 1)
        self._set_if(osd_queue, "max-size-bytes", 0)
        self._set_if(osd_queue, "max-size-time", 0)
        self._set_if(osd_queue, "leaky", 2)

        # dGPU GPU mode keeps rectangle rendering off the CPU. Text stays off;
        # rectangles are the only requirement in this phase.
        self._set_if(osd, "process-mode", 1)
        self._set_if(osd, "display-bbox", True)
        self._set_if(osd, "display-text", False)
        self._set_if(osd, "display-mask", False)
        self._set_if(osd, "gpu-id", 0)

        try:
            self.tiler.unlink(self.wall_queue)
        except Exception:
            pass

        self.osd_queue = osd_queue
        self.osd = osd
        self.pipeline.add(osd_queue)
        self.pipeline.add(osd)
        if not self.tiler.link(osd_queue):
            raise RuntimeError("failed tiler -> detection OSD queue")
        if not osd_queue.link(osd):
            raise RuntimeError("failed detection OSD queue -> nvdsosd")
        if not osd.link(self.wall_queue):
            raise RuntimeError("failed nvdsosd -> wall queue")

        mux_src = self.mux.get_static_pad("src")
        if mux_src is None:
            raise RuntimeError("nvstreammux has no src pad")
        mux_src.add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._inject_latest_boxes_native,
        )

        tiler_src = self.tiler.get_static_pad("src")
        if tiler_src is not None:
            tiler_src.add_probe(
                self.Gst.PadProbeType.BUFFER,
                self._count_tiled_objects_native,
            )

        self.osd_enabled = True
        print(
            "DETECTION_ONLY path ready: camera+YOLO26m only; "
            "native-meta=1 osd=1 tracker=0 reid=0 face=0 heatmap=0; "
            f"wall={self.display_wall_width}x{self.display_wall_height} "
            "mux=20fps-synced sink_sync=0 late_drop=off",
            flush=True,
        )

    def _inject_latest_boxes_native(self, _pad, info):
        bridge = self.meta_bridge
        if bridge is None:
            return self.Gst.PadProbeReturn.OK
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return self.Gst.PadProbeReturn.OK

        now = time.monotonic()
        with self.det_lock:
            snapshots = {
                cid: dict(value)
                for cid, value in self.latest_detections.items()
                if value
            }

        added_this_buffer = 0
        for source_index, cid in enumerate(self.camera_ids):
            snapshot = snapshots.get(cid)
            if not snapshot:
                continue
            captured = float(snapshot.get("captured_mono") or 0.0)
            if captured <= 0.0:
                continue
            if (now - captured) * 1000.0 > self.meta_max_age_ms:
                continue
            boxes = list(snapshot.get("boxes") or [])
            if not boxes:
                continue
            frame_size = snapshot.get("frame_size") or [base.INFER_WIDTH, base.INFER_HEIGHT]
            try:
                added = bridge.add_person_boxes(
                    gst_buffer,
                    source_index,
                    boxes,
                    self.frame_width,
                    self.frame_height,
                    int(frame_size[0]),
                    int(frame_size[1]),
                )
                if added > 0:
                    added_this_buffer += added
            except Exception as exc:
                self.meta_bridge_error = f"runtime bridge: {type(exc).__name__}: {exc}"

        self._boxes_meta += added_this_buffer
        self._bridge_frames += 1
        return self.Gst.PadProbeReturn.OK

    def _count_tiled_objects_native(self, _pad, info):
        now = time.monotonic()
        if now < self._diag_next:
            return self.Gst.PadProbeReturn.OK
        self._diag_next = now + 1.0
        bridge = self.meta_bridge
        gst_buffer = info.get_buffer()
        if bridge is None or gst_buffer is None:
            return self.Gst.PadProbeReturn.OK
        try:
            count = bridge.count_objects(gst_buffer)
            self._tiled_objects_last = max(0, count)
            if count > 0:
                self._tiled_objects_total += count
        except Exception as exc:
            self.meta_bridge_error = f"count bridge: {type(exc).__name__}: {exc}"
        return self.Gst.PadProbeReturn.OK

    def _adapt_detector_rate(self, min_camera_fps: float) -> None:
        # Rate + duty controller. Since the per-batch resolution is now lower,
        # each burst should be shorter; still reduce load immediately if any
        # camera cadence dips.
        super()._adapt_detector_rate(min_camera_fps)
        with self.det_lock:
            if not self.detector_ready:
                return
            if min_camera_fps < 19.3:
                self.max_gpu_duty = max(0.12, self.max_gpu_duty * 0.78)
            elif min_camera_fps >= 19.8:
                self.max_gpu_duty = min(0.18, self.max_gpu_duty + 0.004)

    def _print_stats(self) -> bool:
        result = super()._print_stats()
        print(
            "DETECTION_ONLY "
            f"native_meta={int(self.meta_bridge is not None)} osd={int(self.osd_enabled)} "
            f"boxes_meta={self._boxes_meta} tiled_objects={self._tiled_objects_last} "
            f"bridge_frames={self._bridge_frames} hold={self.meta_max_age_ms:.0f}ms "
            f"duty_cap={self.max_gpu_duty:.0%} wall={self.display_wall_width}x{self.display_wall_height} "
            f"bridge_error={self.meta_bridge_error or '-'} "
            "tracker=0 reid=0 face=0 heatmap=0",
            flush=True,
        )
        return result


def run() -> int:
    return NativeCameraYolo26mDetectionOnly().run()


if __name__ == "__main__":
    raise SystemExit(run())
