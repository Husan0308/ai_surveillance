from __future__ import annotations

import os
import sys
import time

# GTX 1050 Ti smoothness-first profile. Explicit env values still win.
# Keep the proven camera ingest path low-latency and give YOLO a bounded GPU duty.
os.environ.setdefault("AI_WALL_SINK_SYNC", "0")
os.environ.setdefault("AI_YOLO_START_BATCH_FPS", "0.75")
os.environ.setdefault("AI_YOLO_MAX_BATCH_FPS", "1.00")
os.environ.setdefault("AI_YOLO_MIN_BATCH_FPS", "0.45")
os.environ.setdefault("AI_YOLO_MAX_GPU_DUTY", "0.18")
os.environ.setdefault("AI_YOLO_CONF", "0.20")
os.environ.setdefault("AI_CAMERA_FPS_FLOOR", "19.3")
os.environ.setdefault("AI_CAMERA_FPS_GOOD", "19.8")

from . import deepstream_yolo26m_batch6_wall as base


class NativeCameraYolo26mDetectionOnly(base.NativeCameraYolo26mBatch6Wall):
    """Only the proven camera wall plus YOLO26m person detection.

    Camera/display hot path:
        RTSP -> NVDEC -> tee -> latest-only display queue -> nvstreammux
             -> nvmultistreamtiler -> latest-only OSD queue
             -> nvvideoconvert(RGBA/NVMM) -> nvdsosd -> latest wall queue -> EGL

    Detector side path:
        tee -> ticket gate -> one fresh frame per camera -> BGRx appsink
            -> exactly six frames -> one YOLO26m CUDA call

    No tracker, ReID, face, pose, heatmap, API, mmap, JPEG or Qt UI.
    """

    def __init__(self):
        self.pyds = None
        self.osd = None
        self.osd_queue = None
        self.osd_convert = None
        self.osd_caps = None
        self.osd_enabled = False

        # Detector runs below 1 Hz/camera on this GPU. Keep the latest detection
        # visible across detector gaps instead of letting boxes flicker off.
        self.meta_max_age_ms = max(
            500.0,
            float(os.environ.get("AI_DETECTION_BOX_HOLD_MS", "2500")),
        )

        # 3840x1440 was wasteful on the GTX 1050 Ti and is downscaled again by
        # the desktop/AnyDesk. 1920x720 still gives each 3x2 tile 640x360 while
        # cutting tiler/OSD/render pixels by 75 percent.
        self.display_wall_width = max(
            960, int(os.environ.get("AI_DETECTION_WALL_WIDTH", "1920"))
        )
        self.display_wall_height = max(
            360, int(os.environ.get("AI_DETECTION_WALL_HEIGHT", "720"))
        )

        self._boxes_drawn = 0
        self._osd_frames = 0
        self._tiled_objects_last = 0
        self._tiled_objects_total = 0
        self._meta_diag_next = 0.0

        super().__init__()

        # Override only the DISPLAY canvas. nvstreammux remains at camera
        # resolution (1280x720), so detector/source detail is not reduced.
        self._set_if(self.tiler, "width", self.display_wall_width)
        self._set_if(self.tiler, "height", self.display_wall_height)
        self.wall_width = self.display_wall_width
        self.wall_height = self.display_wall_height

        # NVIDIA recommends sync=0 for live RTSP sinks when performance/latency
        # matters. The latest-only queues prevent latency accumulation.
        self._set_if(self.sink, "sync", False)
        self._set_if(self.sink, "qos", False)

        self._setup_detection_osd()

    def _setup_detection_osd(self) -> None:
        try:
            import pyds
            self.pyds = pyds
        except Exception as exc:
            print(
                "DETECTION_ONLY pyds=0; YOLO still runs but bbox rendering is unavailable: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return

        osd_queue = self.Gst.ElementFactory.make("queue", "detection_osd_queue")
        osd_convert = self.Gst.ElementFactory.make("nvvideoconvert", "detection_osd_convert")
        osd_caps = self.Gst.ElementFactory.make("capsfilter", "detection_osd_caps")
        osd = self.Gst.ElementFactory.make("nvdsosd", "detection_osd")
        if None in (osd_queue, osd_convert, osd_caps, osd):
            print(
                "DETECTION_ONLY OSD elements unavailable; YOLO runs but bbox rendering is disabled",
                file=sys.stderr,
                flush=True,
            )
            return

        # Official DeepStream dGPU display examples place nvvideoconvert before
        # nvdsosd. Use RGBA explicitly so OSD negotiation is deterministic.
        osd_caps.set_property(
            "caps",
            self.Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
        )
        self._set_if(osd_convert, "gpu-id", 0)
        self._set_if(osd_convert, "nvbuf-memory-type", 2)

        # OSD is downstream-only work. If it/rendering stalls, drop the OLD
        # tiled frame rather than back-pressure six RTSP decoders.
        self._set_if(osd_queue, "max-size-buffers", 1)
        self._set_if(osd_queue, "max-size-bytes", 0)
        self._set_if(osd_queue, "max-size-time", 0)
        self._set_if(osd_queue, "leaky", 2)

        # CPU OSD + RGBA is the most conservative DeepStream 7.1 path. Only
        # rectangles are enabled; text is deliberately off to keep it light.
        self._set_if(osd, "process-mode", 0)
        self._set_if(osd, "display-bbox", True)
        self._set_if(osd, "display-text", False)
        self._set_if(osd, "display-mask", False)
        self._set_if(osd, "gpu-id", 0)

        try:
            self.tiler.unlink(self.wall_queue)
        except Exception:
            pass

        self.osd_queue = osd_queue
        self.osd_convert = osd_convert
        self.osd_caps = osd_caps
        self.osd = osd
        for element in (osd_queue, osd_convert, osd_caps, osd):
            self.pipeline.add(element)

        if not self.tiler.link(osd_queue):
            raise RuntimeError("failed tiler -> OSD queue")
        if not osd_queue.link(osd_convert):
            raise RuntimeError("failed OSD queue -> nvvideoconvert")
        if not osd_convert.link(osd_caps):
            raise RuntimeError("failed nvvideoconvert -> RGBA caps")
        if not osd_caps.link(osd):
            raise RuntimeError("failed RGBA caps -> nvdsosd")
        if not osd.link(self.wall_queue):
            raise RuntimeError("failed nvdsosd -> wall queue")

        # Add detector object metadata BEFORE tiler. NVIDIA's tiler transforms
        # bbox metadata into the tiled coordinates automatically.
        mux_src = self.mux.get_static_pad("src")
        if mux_src is None:
            raise RuntimeError("nvstreammux has no src pad for detection metadata")
        mux_src.add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._inject_latest_detection_meta,
        )

        # Low-rate diagnostic: prove object meta survives the tiler. This probe
        # only walks metadata about once per second, not on every frame.
        tiler_src = self.tiler.get_static_pad("src")
        if tiler_src is not None:
            tiler_src.add_probe(
                self.Gst.PadProbeType.BUFFER,
                self._verify_tiled_metadata,
            )

        self.osd_enabled = True
        print(
            "DETECTION_ONLY metadata path: pyds=1 osd=1 tracker=0 reid=0 face=0 heatmap=0; "
            f"wall={self.display_wall_width}x{self.display_wall_height} "
            "osd_mode=CPU/RGBA sink_sync=0",
            flush=True,
        )

    def _iter_frames(self, batch_meta):
        node = batch_meta.frame_meta_list
        while node is not None:
            try:
                frame_meta = self.pyds.NvDsFrameMeta.cast(node.data)
            except StopIteration:
                break
            yield frame_meta
            try:
                node = node.next
            except StopIteration:
                break

    def _iter_objects(self, frame_meta):
        node = frame_meta.obj_meta_list
        while node is not None:
            try:
                obj_meta = self.pyds.NvDsObjectMeta.cast(node.data)
            except StopIteration:
                break
            yield obj_meta
            try:
                node = node.next
            except StopIteration:
                break

    def _camera_id(self, frame_meta) -> str | None:
        index = int(getattr(frame_meta, "pad_index", -1))
        if not (0 <= index < len(self.camera_ids)):
            index = int(getattr(frame_meta, "source_id", -1))
        if not (0 <= index < len(self.camera_ids)):
            return None
        return self.camera_ids[index]

    def _add_box_meta(self, batch_meta, frame_meta, item: dict, frame_size) -> None:
        fw, fh = [max(1, int(v)) for v in frame_size]
        sx = float(self.frame_width) / float(fw)
        sy = float(self.frame_height) / float(fh)
        x1, y1, x2, y2 = [float(v) for v in item.get("xyxy", [0, 0, 1, 1])]

        left = max(0.0, min(float(self.frame_width - 1), x1 * sx))
        top = max(0.0, min(float(self.frame_height - 1), y1 * sy))
        right = max(left + 1.0, min(float(self.frame_width), x2 * sx))
        bottom = max(top + 1.0, min(float(self.frame_height), y2 * sy))

        obj = self.pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
        obj.unique_component_id = 1
        obj.class_id = 0
        obj.confidence = float(item.get("confidence") or 0.0)
        obj.object_id = getattr(self.pyds, "UNTRACKED_OBJECT_ID", 0xFFFFFFFFFFFFFFFF)
        try:
            obj.obj_label = "Person"
        except Exception:
            pass

        rect = obj.rect_params
        rect.left = left
        rect.top = top
        rect.width = right - left
        rect.height = bottom - top
        # 4 px at the source plane remains clearly visible after tiling/downscale.
        rect.border_width = 4
        rect.border_color.set(0.0, 1.0, 0.10, 1.0)
        try:
            rect.has_bg_color = 0
        except Exception:
            pass

        try:
            detector = obj.detector_bbox_info.org_bbox_coords
            detector.left = left
            detector.top = top
            detector.width = right - left
            detector.height = bottom - top
        except Exception:
            pass

        self.pyds.nvds_add_obj_meta_to_frame(frame_meta, obj, None)
        self._boxes_drawn += 1

    def _inject_latest_detection_meta(self, _pad, info):
        if self.pyds is None:
            return self.Gst.PadProbeReturn.OK
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return self.Gst.PadProbeReturn.OK
        batch_meta = self.pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return self.Gst.PadProbeReturn.OK

        now = time.monotonic()
        with self.det_lock:
            snapshots = {
                cid: dict(value)
                for cid, value in self.latest_detections.items()
                if value
            }

        for frame_meta in self._iter_frames(batch_meta):
            cid = self._camera_id(frame_meta)
            if cid is None:
                continue
            snapshot = snapshots.get(cid)
            if not snapshot:
                continue
            captured = float(snapshot.get("captured_mono") or 0.0)
            if captured <= 0.0:
                continue
            age_ms = (now - captured) * 1000.0
            if age_ms > self.meta_max_age_ms:
                continue
            frame_size = snapshot.get("frame_size") or [base.INFER_WIDTH, base.INFER_HEIGHT]
            for item in snapshot.get("boxes") or []:
                self._add_box_meta(batch_meta, frame_meta, item, frame_size)
        self._osd_frames += 1
        return self.Gst.PadProbeReturn.OK

    def _verify_tiled_metadata(self, _pad, info):
        now = time.monotonic()
        if now < self._meta_diag_next or self.pyds is None:
            return self.Gst.PadProbeReturn.OK
        self._meta_diag_next = now + 1.0
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return self.Gst.PadProbeReturn.OK
        batch_meta = self.pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            self._tiled_objects_last = 0
            return self.Gst.PadProbeReturn.OK
        count = 0
        for frame_meta in self._iter_frames(batch_meta):
            count += sum(1 for _ in self._iter_objects(frame_meta))
        self._tiled_objects_last = count
        self._tiled_objects_total += count
        return self.Gst.PadProbeReturn.OK

    def _adapt_detector_rate(self, min_camera_fps: float) -> None:
        # Camera cadence is authoritative. Reduce both call-rate and GPU duty
        # quickly when any source falls below the stable ~20 FPS baseline.
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
            f"osd={int(self.osd_enabled)} boxes_meta={self._boxes_drawn} "
            f"tiled_objects={self._tiled_objects_last} osd_frames={self._osd_frames} "
            f"hold={self.meta_max_age_ms:.0f}ms duty_cap={self.max_gpu_duty:.0%} "
            f"wall={self.display_wall_width}x{self.display_wall_height} "
            "tracker=0 reid=0 face=0 heatmap=0",
            flush=True,
        )
        return result


def run() -> int:
    return NativeCameraYolo26mDetectionOnly().run()


if __name__ == "__main__":
    raise SystemExit(run())
