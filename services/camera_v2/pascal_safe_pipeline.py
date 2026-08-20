from __future__ import annotations

"""Pascal-safe RF-DETR camera runtime with a display-first graph.

The production display path is intentionally simple and never waits on the
external detector:

    RTSP/NVDEC -> nvstreammux -> tee -> display queue -> display tiler -> OSD -> sink
                                \\-> analysis queue -> analysis tiler -> appsink

The old detector design used a zero-copy per-source split after nvstreammux. Its
child buffers kept the original mux batch alive until all children were returned.
With sparse gated per-camera queues this could retain several different parent
batches at once and exhaust the mux buffer pool. The hardware log stopped at
seven mux batches with an eight-buffer mux pool, exactly matching that failure
mode. The analysis branch below has no zero-copy per-source split.

It produces one temporary 2x3 analysis wall only when the detector scheduler has
armed a capture request. Each tile is exactly RF-DETR's input size, so Python only
copies the requested tile; it does not resize six camera frames or retain DeepStream
batch buffers.
"""

import os
import time

import numpy as np

from .rfdetr_backend import install as _install_rfdetr_backend

_install_rfdetr_backend()

from .detection import CameraDetectionV2, INFER_HEIGHT, INFER_WIDTH
from .secure import SecureCameraWallV2


class CameraPascalSafeRuntime(CameraDetectionV2):
    """RF-DETR + bounded motion prediction, with no DeepStream tracker."""

    ANALYSIS_COLUMNS = 2
    ANALYSIS_ROWS = 3

    def __init__(self) -> None:
        backend = os.environ.get("CAMERA_V2_DISPLAY_BACKEND", "egl").strip().lower()
        self.display_backend = backend if backend in {"egl", "x11"} else "egl"
        self.display_failover_requested = False
        self.display_watch_started = 0.0
        self.safe_wall_frames = 0
        self.safe_mux_batches = 0
        self.safe_sink_buffers = 0
        self.analysis_frames = 0
        self.source_track_counts: dict[int, int] = {}
        self.tracked_now = 0
        self._analysis_layout_logged = False
        self._startup_stall_reported = False
        super().__init__()
        self.source_track_counts = {
            int(source_id): 0 for source_id in self.camera_index.values()
        }
        self.tracker_backend = "motion-predictor"
        self.tracker = None

    def _preflight(self) -> None:
        super()._preflight()
        for plugin in ("tee", "nvmultistreamtiler", "nvvideoconvert", "appsink"):
            if self.Gst.ElementFactory.find(plugin) is None:
                raise RuntimeError(f"required Pascal-safe plugin is unavailable: {plugin}")
        if self.display_backend == "x11" and self.Gst.ElementFactory.find("ximagesink") is None:
            raise RuntimeError("ximagesink is unavailable for X11 display fallback")

    def _make(self, factory: str, name: str):
        if factory == "nveglglessink" and self.display_backend == "x11":
            factory = "ximagesink"
        return super()._make(factory, name)

    def _add_camera(self, index, camera) -> None:
        """Preserve the proven source -> queue -> nvstreammux ingest path."""

        cid = camera.camera_id
        self.camera_index[cid] = int(index)
        self.capture_requested[cid] = False
        SecureCameraWallV2._add_camera(self, index, camera)

    @staticmethod
    def _queue_latest(owner, element, buffers: int = 2) -> None:
        owner._set_if(element, "max-size-buffers", max(1, int(buffers)))
        owner._set_if(element, "max-size-bytes", 0)
        owner._set_if(element, "max-size-time", 0)
        owner._set_if(element, "leaky", 2)
        owner._set_if(element, "silent", True)

    def _request_src_pad(self, element, name: str):
        request_simple = getattr(element, "request_pad_simple", None)
        pad = request_simple(name) if request_simple else None
        if pad is None:
            pad = element.get_request_pad(name)
        if pad is None:
            raise RuntimeError(f"{element.get_name()} could not allocate {name}")
        self.tee_request_pads.append((element, pad))
        return pad

    def _analysis_gate_probe(self, _pad, _info):
        """Drop analysis batches immediately unless RF-DETR requested a sample."""

        with self.capture_lock:
            requested = any(bool(v) for v in self.capture_requested.values())
        return self.Gst.PadProbeReturn.OK if requested else self.Gst.PadProbeReturn.DROP

    def _install_analysis_inference(self) -> None:
        """Attach a non-retaining detector branch after nvstreammux.

        Both branches consume the same batched mux buffer, but the detector branch
        immediately drops unrequested batches. Requested batches are fully consumed
        by a second tiler, which creates its own 2x3 output surface. No retained
        per-source child buffers exist, so the mux buffer can always return to its
        pool after the analysis wall is produced.
        """

        mux_src = self.mux.get_static_pad("src")
        display_tiler_sink = self.tiler.get_static_pad("sink")
        if mux_src is None or display_tiler_sink is None:
            raise RuntimeError("could not inspect nvstreammux -> display tiler link")
        if mux_src.is_linked():
            self.mux.unlink(self.tiler)
        if mux_src.is_linked() or display_tiler_sink.is_linked():
            raise RuntimeError("could not detach nvstreammux -> display tiler")

        tee = self._make("tee", "pascal_mux_tee")
        display_q = self._make("queue", "pascal_display_branch")
        analysis_q = self._make("queue", "pascal_analysis_branch")
        analysis_tiler = self._make("nvmultistreamtiler", "pascal_analysis_tiler")
        analysis_convert = self._make("nvvideoconvert", "pascal_analysis_convert")
        analysis_caps = self._make("capsfilter", "pascal_analysis_caps")
        analysis_sink = self._make("appsink", "pascal_analysis_sink")

        self._queue_latest(self, display_q, 2)
        self._queue_latest(self, analysis_q, 1)

        analysis_width = INFER_WIDTH * self.ANALYSIS_COLUMNS
        analysis_height = INFER_HEIGHT * self.ANALYSIS_ROWS
        self._set_if(analysis_tiler, "rows", self.ANALYSIS_ROWS)
        self._set_if(analysis_tiler, "columns", self.ANALYSIS_COLUMNS)
        self._set_if(analysis_tiler, "width", analysis_width)
        self._set_if(analysis_tiler, "height", analysis_height)
        self._set_if(analysis_tiler, "gpu-id", self.gpu_id)
        self._set_if(analysis_tiler, "nvbuf-memory-type", 2)
        self._set_if(analysis_tiler, "compute-hw", 1)
        self._set_if(analysis_tiler, "interpolation-method", 2)
        if analysis_tiler.find_property("show-source") is not None:
            analysis_tiler.set_property("show-source", -1)

        self._set_if(analysis_convert, "gpu-id", self.gpu_id)
        self._set_if(analysis_convert, "compute-hw", 1)
        analysis_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                "video/x-raw,format=BGRx,"
                f"width={analysis_width},height={analysis_height},pixel-aspect-ratio=1/1"
            ),
        )
        analysis_sink.set_property("emit-signals", True)
        analysis_sink.set_property("sync", False)
        analysis_sink.set_property("drop", True)
        analysis_sink.set_property("max-buffers", 1)
        self._set_if(analysis_sink, "enable-last-sample", False)
        self._set_if(analysis_sink, "wait-on-eos", False)

        for element in (
            tee,
            display_q,
            analysis_q,
            analysis_tiler,
            analysis_convert,
            analysis_caps,
            analysis_sink,
        ):
            self.pipeline.add(element)

        if not self.mux.link(tee):
            raise RuntimeError("failed nvstreammux -> detector/display tee")

        tee_display = self._request_src_pad(tee, "src_%u")
        tee_analysis = self._request_src_pad(tee, "src_%u")
        if tee_display.link(display_q.get_static_pad("sink")) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError("failed mux tee -> display queue")
        if tee_analysis.link(analysis_q.get_static_pad("sink")) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError("failed mux tee -> analysis queue")
        if not display_q.link(self.tiler):
            raise RuntimeError("failed display queue -> display tiler")
        if not analysis_q.link(analysis_tiler):
            raise RuntimeError("failed analysis queue -> analysis tiler")
        if not analysis_tiler.link(analysis_convert):
            raise RuntimeError("failed analysis tiler -> analysis convert")
        if not analysis_convert.link(analysis_caps):
            raise RuntimeError("failed analysis convert -> BGRx caps")
        if not analysis_caps.link(analysis_sink):
            raise RuntimeError("failed analysis caps -> appsink")

        # BUFFER-only gate: CAPS/SEGMENT events still pass, so negotiation is
        # complete before the first detector sample is requested.
        analysis_q.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._analysis_gate_probe,
        )
        analysis_sink.connect("new-sample", self._on_analysis_sample)

        self.postmux_tee = tee
        self.postmux_display_queue = display_q
        self.analysis_queue = analysis_q
        self.analysis_tiler = analysis_tiler
        self.analysis_convert = analysis_convert
        self.analysis_caps = analysis_caps
        self.analysis_sink = analysis_sink

        print(
            "CAMERA_DETECT_PATH mode=analysis-tiler "
            "source_path=direct-to-nvstreammux demux=disabled "
            "mux_batch_retention=bounded",
            flush=True,
        )

    def _on_analysis_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return self.Gst.FlowReturn.OK

        with self.capture_lock:
            requested = [cid for cid, armed in self.capture_requested.items() if armed]
        if not requested:
            return self.Gst.FlowReturn.OK

        structure = sample.get_caps().get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        expected_width = INFER_WIDTH * self.ANALYSIS_COLUMNS
        expected_height = INFER_HEIGHT * self.ANALYSIS_ROWS
        if (width, height) != (expected_width, expected_height):
            with self.det_lock:
                self.det_error = (
                    f"analysis wall geometry {width}x{height} != "
                    f"{expected_width}x{expected_height}"
                )
            return self.Gst.FlowReturn.OK

        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            return self.Gst.FlowReturn.OK

        captured = time.monotonic()
        delivered: list[str] = []
        try:
            tight_stride = width * 4
            mapped_size = int(getattr(mapped, "size", len(mapped.data)))
            if mapped_size < tight_stride * height:
                raise RuntimeError(
                    f"analysis BGRx buffer too small: {mapped_size} < {tight_stride * height}"
                )
            row_stride = (
                mapped_size // height
                if height > 0 and mapped_size % height == 0
                else tight_stride
            )
            if row_stride < tight_stride:
                raise RuntimeError(
                    f"analysis invalid BGRx stride={row_stride}, tight={tight_stride}"
                )

            raw = np.frombuffer(
                mapped.data,
                dtype=np.uint8,
                count=row_stride * height,
            )
            rows = raw.reshape((height, row_stride))
            bgrx = rows[:, :tight_stride].reshape((height, width, 4))

            for cid in requested:
                index = int(self.camera_index[cid])
                row = index // self.ANALYSIS_COLUMNS
                column = index % self.ANALYSIS_COLUMNS
                y1 = row * INFER_HEIGHT
                y2 = y1 + INFER_HEIGHT
                x1 = column * INFER_WIDTH
                x2 = x1 + INFER_WIDTH
                frame = bgrx[y1:y2, x1:x2, :3].copy()
                if frame.shape != (INFER_HEIGHT, INFER_WIDTH, 3):
                    continue
                self.mailbox.put(cid, captured, frame)
                delivered.append(cid)

            if not self._analysis_layout_logged:
                self._analysis_layout_logged = True
                print(
                    "CAMERA_INFER_LAYOUT "
                    f"wall={width}x{height} stride={row_stride} "
                    f"tile={INFER_WIDTH}x{INFER_HEIGHT} grid="
                    f"{self.ANALYSIS_COLUMNS}x{self.ANALYSIS_ROWS}",
                    flush=True,
                )
        finally:
            buffer.unmap(mapped)

        if delivered:
            with self.capture_lock:
                for cid in delivered:
                    self.capture_requested[cid] = False
            self.analysis_frames += 1
        return self.Gst.FlowReturn.OK

    def _install_osd_and_meta(self) -> None:
        self._install_analysis_inference()

        queue_src = self.wall_queue.get_static_pad("src")
        sink_pad = self.sink.get_static_pad("sink")
        if queue_src is None or sink_pad is None:
            raise RuntimeError("could not inspect baseline wall -> display pads")
        if queue_src.is_linked():
            self.wall_queue.unlink(self.sink)
        if queue_src.is_linked() or sink_pad.is_linked():
            raise RuntimeError("could not detach baseline wall -> display link")

        convert = self._make("nvvideoconvert", "pascal_wall_convert")
        caps = self._make("capsfilter", "pascal_wall_caps")
        osd = self._make("nvdsosd", "pascal_osd")
        self._set_if(convert, "gpu-id", self.gpu_id)
        self._set_if(convert, "compute-hw", 1)
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

        if self.display_backend == "egl":
            if not osd.link(self.sink):
                raise RuntimeError("failed nvdsosd -> nveglglessink")
        else:
            download = self._make("nvvideoconvert", "pascal_x11_download")
            sys_caps = self._make("capsfilter", "pascal_x11_caps")
            self._set_if(download, "gpu-id", self.gpu_id)
            self._set_if(download, "compute-hw", 1)
            sys_caps.set_property(
                "caps",
                self.Gst.Caps.from_string("video/x-raw,format=BGRx"),
            )
            self.pipeline.add(download)
            self.pipeline.add(sys_caps)
            if not osd.link(download):
                raise RuntimeError("failed nvdsosd -> X11 download convert")
            if not download.link(sys_caps):
                raise RuntimeError("failed X11 download convert -> system caps")
            if not sys_caps.link(self.sink):
                raise RuntimeError("failed system BGRx -> ximagesink")
            self.x11_download = download
            self.x11_caps = sys_caps

        mux_src = self.mux.get_static_pad("src")
        osd_src = osd.get_static_pad("src")
        final_sink_pad = self.sink.get_static_pad("sink")
        if mux_src is None or osd_src is None or final_sink_pad is None:
            raise RuntimeError("could not obtain mux/OSD/display probe pads")
        mux_src.add_probe(self.Gst.PadProbeType.BUFFER, self._pascal_mux_probe)
        osd_src.add_probe(self.Gst.PadProbeType.BUFFER, self._pascal_wall_probe)
        final_sink_pad.add_probe(self.Gst.PadProbeType.BUFFER, self._pascal_sink_probe)
        self.osd = osd

    def _pascal_mux_probe(self, pad, info):
        self.safe_mux_batches += 1
        return CameraDetectionV2._inject_boxes_probe(self, pad, info)

    def _pascal_wall_probe(self, pad, info):
        self.safe_wall_frames += 1
        return CameraDetectionV2._wall_probe(self, pad, info)

    def _pascal_sink_probe(self, _pad, _info):
        self.safe_sink_buffers += 1
        return self.Gst.PadProbeReturn.OK

    def _active_motion_counts(self) -> dict[int, int]:
        now = time.monotonic()
        output = {int(source_id): 0 for source_id in self.camera_index.values()}
        boxes = getattr(self, "boxes", None)
        if boxes is None:
            return output
        with boxes.lock:
            for cid, source_id in self.camera_index.items():
                active = sum(
                    1
                    for track in boxes.tracks.get(cid, {}).values()
                    if now - float(track.last_det_t) <= float(boxes.max_age)
                )
                output[int(source_id)] = active
        return output

    def live_source_counts(self) -> dict[int, int]:
        counts = self._active_motion_counts()
        with self.det_lock:
            self.source_track_counts = counts
            self.tracked_now = sum(counts.values())
        return dict(counts)

    def _display_watchdog(self) -> bool:
        if self.display_backend != "egl" or self._stopping:
            return False
        elapsed = time.monotonic() - self.display_watch_started
        if elapsed < float(os.environ.get("CAMERA_V2_EGL_FAILOVER_SEC", "8.0")):
            return True

        source_frames = sum(int(stat.frames) for stat in self.stats.values())
        rendered, _dropped = self._sink_stats()
        if (
            source_frames >= 60
            and self.safe_mux_batches >= 10
            and self.safe_wall_frames >= 10
            and self.safe_sink_buffers >= 10
            and rendered == 0
        ):
            self.display_failover_requested = True
            print(
                "CAMERA_DISPLAY_FAILOVER "
                f"from=egl to=x11 reason=zero-render source_frames={source_frames} "
                f"mux_batches={self.safe_mux_batches} wall_frames={self.safe_wall_frames} "
                f"sink_buffers={self.safe_sink_buffers}",
                flush=True,
            )
            self.stop()
            return False
        return True

    def _startup_watchdog(self) -> bool:
        if self._stopping:
            return False
        if time.monotonic() - self.display_watch_started < 10.0:
            return True

        source_total = sum(int(stat.frames) for stat in self.stats.values())
        if source_total == 0:
            stage = "source-or-auth"
        elif self.safe_mux_batches == 0:
            stage = "nvstreammux"
        elif self.safe_wall_frames == 0:
            stage = "display-tiler-or-osd"
        elif self.safe_sink_buffers == 0:
            stage = "display-sink-link"
        else:
            return False

        if not self._startup_stall_reported:
            self._startup_stall_reported = True
            print(
                "CAMERA_STARTUP_STALL "
                f"stage={stage} source_frames={source_total} "
                f"mux_batches={self.safe_mux_batches} wall_frames={self.safe_wall_frames} "
                f"sink_buffers={self.safe_sink_buffers}",
                flush=True,
            )
        return True

    def _print_stats(self) -> bool:
        keep = CameraDetectionV2._print_stats(self)
        counts = self.live_source_counts()
        rendered, dropped = self._sink_stats()
        source_total = sum(int(stat.frames) for stat in self.stats.values())
        print(
            "CAMERA_PASCAL_SAFE "
            f"display={self.display_backend} source_frames={source_total} "
            f"mux_batches={self.safe_mux_batches} wall_frames={self.safe_wall_frames} "
            f"sink_buffers={self.safe_sink_buffers} analysis_frames={self.analysis_frames} "
            f"tracked_now={self.tracked_now} source_counts={counts} "
            f"rendered={rendered if rendered is not None else '?'} "
            f"dropped={dropped if dropped is not None else '?'} "
            "nvtracker=0 tracker=motion-predictor detector_path=analysis-tiler",
            flush=True,
        )
        return keep

    def run(self) -> int:
        self.display_watch_started = time.monotonic()
        self.GLib.timeout_add(1000, self._startup_watchdog)
        if self.display_backend == "egl":
            self.GLib.timeout_add(1000, self._display_watchdog)
        print(
            "CAMERA_PASCAL_SAFE ready backend=RF-DETR-S "
            f"display={self.display_backend} tracker=motion-predictor nvtracker=disabled "
            "source_path=direct-to-mux detector_path=analysis-tiler demux=disabled",
            flush=True,
        )
        return super().run()


def main() -> int:
    enabled = os.environ.get("CAMERA_V2_PASCAL_SAFE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled:
        raise RuntimeError("CAMERA_V2_PASCAL_SAFE=1 is required")
    return CameraPascalSafeRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
