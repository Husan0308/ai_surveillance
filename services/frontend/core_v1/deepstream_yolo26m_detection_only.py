from __future__ import annotations

import os
import sys
import time

# GTX 1050 Ti: protect camera cadence first. Explicit env values still win.
os.environ.setdefault("AI_YOLO_START_BATCH_FPS", "0.90")
os.environ.setdefault("AI_YOLO_MAX_BATCH_FPS", "1.25")
os.environ.setdefault("AI_YOLO_MIN_BATCH_FPS", "0.50")
os.environ.setdefault("AI_YOLO_MAX_GPU_DUTY", "0.20")
os.environ.setdefault("AI_YOLO_CONF", "0.20")
os.environ.setdefault("AI_CAMERA_FPS_FLOOR", "19.2")
os.environ.setdefault("AI_CAMERA_FPS_GOOD", "19.8")

from . import deepstream_yolo26m_batch6_wall as base


class NativeCameraYolo26mDetectionOnly(base.NativeCameraYolo26mBatch6Wall):
    """Camera + YOLO26m only.

    Display hot path stays GPU-native:
        RTSP -> NVDEC -> tee -> latest-only queue -> nvstreammux
             -> tiler -> GPU nvdsosd -> EGL

    Detection stays on the side branch:
        tee -> ticket gate -> one fresh frame/camera -> BGRx appsink
            -> exactly 6 frames -> one YOLO26m CUDA call

    There is intentionally NO tracker, ReID, face recognition, pose, heatmap,
    API, mmap, JPEG, or Qt UI in this runtime.
    """

    def __init__(self):
        self.pyds = None
        self.osd = None
        self.osd_enabled = False
        self.meta_max_age_ms = max(
            250.0,
            float(os.environ.get("AI_DETECTION_BOX_HOLD_MS", "1200")),
        )
        self._boxes_drawn = 0
        self._osd_frames = 0
        super().__init__()
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

        osd = self.Gst.ElementFactory.make("nvdsosd", "detection_osd")
        if osd is None:
            print(
                "DETECTION_ONLY nvdsosd=0; YOLO still runs but bbox rendering is unavailable",
                file=sys.stderr,
                flush=True,
            )
            return

        # Rewire before PLAYING: tiler -> GPU OSD -> latest-only wall queue.
        try:
            self.tiler.unlink(self.wall_queue)
        except Exception:
            pass

        self.osd = osd
        self.pipeline.add(osd)
        self._set_if(osd, "process-mode", 1)  # GPU mode
        self._set_if(osd, "display-bbox", True)
        self._set_if(osd, "display-text", True)
        self._set_if(osd, "display-mask", False)
        self._set_if(osd, "gpu-id", 0)

        if not self.tiler.link(osd) or not osd.link(self.wall_queue):
            raise RuntimeError("failed to link tiler -> nvdsosd -> wall queue")

        mux_src = self.mux.get_static_pad("src")
        if mux_src is None:
            raise RuntimeError("nvstreammux has no src pad for detection metadata")
        mux_src.add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._inject_latest_detection_meta,
        )
        self.osd_enabled = True
        print(
            "DETECTION_ONLY metadata path: pyds=1 osd=1 tracker=0 reid=0 face=0 heatmap=0",
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
        rect.border_width = 2
        rect.border_color.set(0.0, 1.0, 0.15, 1.0)
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

        try:
            text = obj.text_params
            text.display_text = f"Person {obj.confidence:.2f}"
            text.x_offset = max(0, int(left))
            text.y_offset = max(0, int(top) - 18)
            text.font_params.font_name = "Sans"
            text.font_params.font_size = 11
            text.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            text.set_bg_clr = 1
            text.text_bg_clr.set(0.0, 0.0, 0.0, 0.55)
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

    def _adapt_detector_rate(self, min_camera_fps: float) -> None:
        # Base controller adjusts batch-rate cap. Add a second guard for GPU duty
        # so camera smoothness wins immediately when FPS dips.
        super()._adapt_detector_rate(min_camera_fps)
        with self.det_lock:
            if not self.detector_ready:
                return
            if min_camera_fps < 19.2:
                self.max_gpu_duty = max(0.14, self.max_gpu_duty * 0.80)
            elif min_camera_fps >= 19.8:
                self.max_gpu_duty = min(0.20, self.max_gpu_duty + 0.005)

    def _print_stats(self) -> bool:
        result = super()._print_stats()
        print(
            "DETECTION_ONLY "
            f"osd={int(self.osd_enabled)} boxes_drawn={self._boxes_drawn} "
            f"osd_frames={self._osd_frames} duty_cap={self.max_gpu_duty:.0%} "
            "tracker=0 reid=0 face=0 heatmap=0",
            flush=True,
        )
        return result


def run() -> int:
    return NativeCameraYolo26mDetectionOnly().run()


if __name__ == "__main__":
    raise SystemExit(run())
