from __future__ import annotations

import os
import sys
import threading
import time

# Intel iGPU owns display/decode/composition; NVIDIA owns YOLO26m only.
# This removes the visible graphics-vs-CUDA contention on GTX 1050 Ti (GP107).
os.environ.setdefault("LIBVA_DRIVER_NAME", "iHD")
os.environ.setdefault("AI_YOLO_INFER_WIDTH", "576")
os.environ.setdefault("AI_YOLO_INFER_HEIGHT", "320")
os.environ.setdefault("AI_YOLO_PREDICT_WIDTH", "576")
os.environ.setdefault("AI_YOLO_PREDICT_HEIGHT", "320")
os.environ.setdefault("AI_YOLO_START_BATCH_FPS", "0.75")
os.environ.setdefault("AI_YOLO_MAX_BATCH_FPS", "1.00")
os.environ.setdefault("AI_YOLO_MIN_BATCH_FPS", "0.45")
os.environ.setdefault("AI_YOLO_MAX_GPU_DUTY", "0.18")
os.environ.setdefault("AI_YOLO_CONF", "0.18")
os.environ.setdefault("AI_WALL_SINK_SYNC", "0")

from shared.config import camera_config
from services.ml_service.cameras.gstreamer import _gstreamer, authenticated_source
from . import deepstream_yolo26m_batch6_wall as base


class HeadlessYolo26mBatch6(base.NativeCameraYolo26mBatch6Wall):
    """NVIDIA-only detector pipeline with no visible rendering work.

    The existing six NVDEC sources and ticketed inference branch are retained,
    but the display branch is dropped before nvstreammux and EGL is replaced by
    fakesink. This makes the NVIDIA GPU responsible only for decode + YOLO26m.
    """

    def __init__(self):
        super().__init__()

        # Keep source-FPS probes working, then drop the display copy before mux.
        for cid in self.camera_ids:
            pad = self.queues[cid].get_static_pad("src")
            if pad is not None:
                pad.add_probe(self.Gst.PadProbeType.BUFFER, self._drop_display_buffer)

        # Do not create an NVIDIA EGL/OpenGL window in detector-only mode.
        try:
            self.wall_queue.unlink(self.sink)
        except Exception:
            pass
        try:
            self.pipeline.remove(self.sink)
        except Exception:
            pass

        fake = self._make("fakesink", "detector_fakesink")
        self._set_if(fake, "sync", False)
        self._set_if(fake, "async", False)
        self._set_if(fake, "qos", False)
        self.pipeline.add(fake)
        if not self.wall_queue.link(fake):
            raise RuntimeError("failed to link detector fakesink")
        self.sink = fake

        # These receive no buffers after the drop probe, but keep them tiny in
        # case a buffer slips through while the pipeline changes state.
        self._set_if(self.mux, "width", 64)
        self._set_if(self.mux, "height", 64)
        self._set_if(self.mux, "sync-inputs", False)
        self._set_if(self.mux, "batched-push-timeout", 10000)
        self._set_if(self.tiler, "width", 64)
        self._set_if(self.tiler, "height", 64)

    def _drop_display_buffer(self, _pad, _info):
        return self.Gst.PadProbeReturn.DROP


class IntelVaDisplayWall:
    """Six-camera smooth wall isolated from NVIDIA CUDA compute.

    Preferred path:
      RTSP -> H264/H265 depay/parse -> Intel VA decode -> VA compositor
           -> one final system-memory download -> Cairo bbox overlay -> X11 sink

    The detector runs in a separate pipeline/thread and only shares the latest
    box coordinates through a lock. No video frame crosses from NVIDIA to Intel.
    """

    def __init__(self, detector: HeadlessYolo26mBatch6):
        Gst = _gstreamer()
        from gi.repository import GLib

        self.Gst = Gst
        self.GLib = GLib
        self.detector = detector
        self.pipeline = Gst.Pipeline.new("intel-va-camera-wall")
        if self.pipeline is None:
            raise RuntimeError("failed to create Intel VA display pipeline")

        self.cameras = [
            dict(item)
            for item in camera_config().get("cameras", [])
            if item.get("online", True)
        ]
        if len(self.cameras) != 6:
            raise RuntimeError(f"dual-GPU wall requires 6 cameras, found {len(self.cameras)}")

        self.camera_ids = [str(item["id"]) for item in self.cameras]
        self.wall_width = max(960, int(os.environ.get("AI_INTEL_WALL_WIDTH", "1920")))
        self.wall_height = max(360, int(os.environ.get("AI_INTEL_WALL_HEIGHT", "720")))
        self.tile_width = self.wall_width // 3
        self.tile_height = self.wall_height // 2
        self.rtsp_latency_ms = max(40, int(os.environ.get("AI_INTEL_RTSP_LATENCY_MS", "100")))
        self.box_hold_ms = max(700.0, float(os.environ.get("AI_DETECTION_BOX_HOLD_MS", "3000")))

        self.sources = {}
        self.decoders = {}
        self._request_pads = []
        self._draw_frames = 0
        self._draw_last = 0
        self._draw_last_t = time.monotonic()

        self.use_va_compositor = Gst.ElementFactory.find("vacompositor") is not None
        compositor_name = "vacompositor" if self.use_va_compositor else "compositor"
        self.compositor = self._make(compositor_name, "intel_compositor")
        self._set_if(self.compositor, "ignore-inactive-pads", True)
        self._set_if(self.compositor, "background", 1)  # black when supported

        self.pipeline.add(self.compositor)

        for index, camera in enumerate(self.cameras):
            self._add_source(index, camera)

        # Final composite conversion/download happens once per wall frame, not
        # once per camera. Cairo then draws only a few rectangle strokes.
        if self.use_va_compositor:
            post = self._make("vapostproc", "intel_wall_post")
        else:
            post = self._make("videoconvert", "intel_wall_post")
        caps = self._make("capsfilter", "intel_wall_caps")
        overlay = self._make("cairooverlay", "intel_bbox_overlay")
        out_q = self._make("queue", "intel_wall_queue")
        sink = self._choose_sink()

        caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,width={self.wall_width},height={self.wall_height}"
            ),
        )

        self._set_if(out_q, "max-size-buffers", 1)
        self._set_if(out_q, "max-size-bytes", 0)
        self._set_if(out_q, "max-size-time", 0)
        self._set_if(out_q, "leaky", 2)
        self._set_if(sink, "sync", False)
        self._set_if(sink, "qos", False)
        self._set_if(sink, "async", False)
        self._set_if(sink, "enable-last-sample", False)

        for element in (post, caps, overlay, out_q, sink):
            self.pipeline.add(element)

        if not self.compositor.link(post):
            raise RuntimeError(f"failed {compositor_name} -> postproc")
        if not post.link(caps):
            raise RuntimeError("failed postproc -> wall caps")
        if not caps.link(overlay):
            raise RuntimeError("failed wall caps -> cairooverlay")
        if not overlay.link(out_q):
            raise RuntimeError("failed cairooverlay -> output queue")
        if not out_q.link(sink):
            raise RuntimeError("failed output queue -> display sink")

        overlay.connect("draw", self._draw_overlay)

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)
        self.loop = GLib.MainLoop()
        GLib.timeout_add_seconds(5, self._print_stats)

        self.sink = sink
        self.overlay = overlay
        self.post = post

    def _make(self, factory: str, name: str):
        element = self.Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"missing GStreamer element: {factory}")
        return element

    @staticmethod
    def _set_if(element, name: str, value) -> bool:
        if element.find_property(name) is None:
            return False
        element.set_property(name, value)
        return True

    def _choose_sink(self):
        session = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session != "wayland" and self.Gst.ElementFactory.find("ximagesink") is not None:
            return self._make("ximagesink", "intel_wall_sink")
        if self.Gst.ElementFactory.find("waylandsink") is not None:
            return self._make("waylandsink", "intel_wall_sink")
        return self._make("autovideosink", "intel_wall_sink")

    def _request_compositor_pad(self, index: int):
        request_simple = getattr(self.compositor, "request_pad_simple", None)
        pad = request_simple("sink_%u") if request_simple else None
        if pad is None:
            pad = self.compositor.get_request_pad("sink_%u")
        if pad is None:
            raise RuntimeError("compositor could not allocate sink pad")
        row, col = divmod(index, 3)
        for key, value in (
            ("xpos", col * self.tile_width),
            ("ypos", row * self.tile_height),
            ("width", self.tile_width),
            ("height", self.tile_height),
        ):
            if pad.find_property(key) is not None:
                pad.set_property(key, value)
        self._request_pads.append(pad)
        return pad

    def _add_source(self, index: int, camera: dict) -> None:
        cid = str(camera["id"])
        codec = str(camera.get("display_codec") or camera.get("codec") or "h264").lower()
        if codec not in {"h264", "h265"}:
            raise RuntimeError(f"{cid}: unsupported display codec {codec}")

        uri = authenticated_source(
            {**camera, "source": camera.get("display_source") or camera.get("source")}
        )
        source = self._make("rtspsrc", f"intel_rtsp_{index}")
        depay = self._make(f"rtp{codec}depay", f"intel_depay_{index}")
        parser = self._make(f"{codec}parse", f"intel_parse_{index}")
        decoder = self._make(f"va{codec}dec", f"intel_decode_{index}")
        queue = self._make("queue", f"intel_queue_{index}")

        source.set_property("location", uri)
        self._set_if(source, "latency", self.rtsp_latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "tcp-timeout", 5_000_000)

        self._set_if(queue, "max-size-buffers", 2)
        self._set_if(queue, "max-size-bytes", 0)
        self._set_if(queue, "max-size-time", 0)
        self._set_if(queue, "leaky", 2)

        for element in (source, depay, parser, decoder, queue):
            self.pipeline.add(element)

        if not depay.link(parser):
            raise RuntimeError(f"{cid}: depay -> parser failed")
        if not parser.link(decoder):
            raise RuntimeError(f"{cid}: parser -> VA decoder failed")
        if not decoder.link(queue):
            raise RuntimeError(f"{cid}: VA decoder -> queue failed")

        comp_pad = self._request_compositor_pad(index)
        qsrc = queue.get_static_pad("src")
        if qsrc.link(comp_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: queue -> compositor failed")

        source.connect("pad-added", self._on_rtsp_pad_added, depay, codec, cid)
        self.sources[cid] = source
        self.decoders[cid] = decoder

    def _on_rtsp_pad_added(self, _source, pad, depay, codec: str, cid: str) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        structure = caps.get_structure(0)
        if str(structure.get_name()) != "application/x-rtp":
            return
        media = str(structure.get_string("media") or "").lower()
        encoding = str(structure.get_string("encoding-name") or "").lower()
        if media != "video" or codec not in encoding:
            return
        sinkpad = depay.get_static_pad("sink")
        if sinkpad.is_linked():
            return
        result = pad.link(sinkpad)
        if result != self.Gst.PadLinkReturn.OK:
            print(f"INTEL_DISPLAY {cid} RTSP -> depay failed: {result}", flush=True)

    def _draw_overlay(self, _overlay, cr, _timestamp, _duration) -> None:
        self._draw_frames += 1
        now = time.monotonic()
        with self.detector.det_lock:
            snapshots = {
                cid: dict(value)
                for cid, value in self.detector.latest_detections.items()
                if value
            }

        for index, cid in enumerate(self.camera_ids):
            snapshot = snapshots.get(cid)
            if not snapshot:
                continue
            captured = float(snapshot.get("captured_mono") or 0.0)
            if captured <= 0.0 or (now - captured) * 1000.0 > self.box_hold_ms:
                continue
            frame_size = snapshot.get("frame_size") or [576, 320]
            fw = max(1.0, float(frame_size[0]))
            fh = max(1.0, float(frame_size[1]))
            row, col = divmod(index, 3)
            ox = col * self.tile_width
            oy = row * self.tile_height
            sx = self.tile_width / fw
            sy = self.tile_height / fh

            for item in snapshot.get("boxes") or []:
                x1, y1, x2, y2 = [float(v) for v in item.get("xyxy", [0, 0, 1, 1])]
                left = ox + max(0.0, min(fw, x1)) * sx
                top = oy + max(0.0, min(fh, y1)) * sy
                width = max(1.0, (min(fw, x2) - max(0.0, x1)) * sx)
                height = max(1.0, (min(fh, y2) - max(0.0, y1)) * sy)
                try:
                    cr.set_source_rgba(0.0, 1.0, 0.10, 1.0)
                    cr.set_line_width(3.0)
                    cr.rectangle(left, top, width, height)
                    cr.stroke()
                except Exception:
                    return

    def _print_stats(self) -> bool:
        now = time.monotonic()
        elapsed = max(0.001, now - self._draw_last_t)
        fps = (self._draw_frames - self._draw_last) / elapsed
        self._draw_last = self._draw_frames
        self._draw_last_t = now
        devices = []
        for cid, decoder in self.decoders.items():
            path = "?"
            if decoder.find_property("device-path") is not None:
                try:
                    path = str(decoder.get_property("device-path") or "?")
                except Exception:
                    pass
            devices.append(f"{cid}:{path}")
        with self.detector.det_lock:
            counts = dict(self.detector.last_counts)
        count_text = " ".join(f"{cid}:{counts.get(cid, 0)}" for cid in self.camera_ids)
        print(
            f"INTEL_DISPLAY fps={fps:.1f} wall={self.wall_width}x{self.wall_height} "
            f"va_compositor={int(self.use_va_compositor)} boxes=[{count_text}] "
            f"devices=[{' '.join(devices)}]",
            flush=True,
        )
        return True

    def _on_bus_message(self, _bus, message) -> None:
        if message.type == self.Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            src = message.src.get_name() if message.src else "unknown"
            print(
                f"INTEL_DISPLAY ERROR source={src} message={err.message} debug={debug or ''}",
                file=sys.stderr,
                flush=True,
            )
            self.loop.quit()
        elif message.type == self.Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            src = message.src.get_name() if message.src else "unknown"
            print(
                f"INTEL_DISPLAY WARNING source={src} message={err.message} debug={debug or ''}",
                flush=True,
            )
        elif message.type == self.Gst.MessageType.EOS:
            self.loop.quit()

    def run(self) -> int:
        result = self.pipeline.set_state(self.Gst.State.PLAYING)
        if result == self.Gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(self.Gst.State.NULL)
            raise RuntimeError("Intel VA display pipeline failed to PLAY")
        print(
            "DUAL_GPU display=Intel-VA detector=NVIDIA-YOLO26m batch=6; "
            f"wall={self.wall_width}x{self.wall_height} sink={self.sink.get_factory().get_name()} "
            f"rtsp_latency={self.rtsp_latency_ms}ms",
            flush=True,
        )
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.set_state(self.Gst.State.NULL)
        return 0


def _missing_dual_gpu_plugins() -> list[str]:
    Gst = _gstreamer()
    required = [
        "rtspsrc",
        "rtph264depay",
        "rtph265depay",
        "h264parse",
        "h265parse",
        "vah264dec",
        "vah265dec",
        "cairooverlay",
    ]
    if Gst.ElementFactory.find("vacompositor") is not None:
        required.append("vapostproc")
    else:
        required.extend(["compositor", "videoconvert"])
    if (
        Gst.ElementFactory.find("ximagesink") is None
        and Gst.ElementFactory.find("waylandsink") is None
        and Gst.ElementFactory.find("autovideosink") is None
    ):
        required.append("ximagesink|waylandsink|autovideosink")
    return [name for name in required if "|" not in name and Gst.ElementFactory.find(name) is None]


def run() -> int:
    missing = _missing_dual_gpu_plugins()
    if missing:
        print(
            "DUAL_GPU unavailable; missing GStreamer plugins: " + ", ".join(missing),
            file=sys.stderr,
            flush=True,
        )
        print(
            "DUAL_GPU falling back to single-NVIDIA camera+detection mode",
            flush=True,
        )
        from . import deepstream_yolo26m_detection_only as fallback
        return fallback.run()

    detector = HeadlessYolo26mBatch6()
    display = IntelVaDisplayWall(detector)
    detector_thread = threading.Thread(
        target=detector.run,
        name="nvidia-yolo26m-detector",
        daemon=True,
    )
    detector_thread.start()

    try:
        return display.run()
    finally:
        detector.stop_event.set()
        detector._clear_capture_requests()
        if detector.latest is not None:
            detector.latest.close()
        try:
            detector.loop.quit()
        except Exception:
            pass
        detector_thread.join(timeout=3.0)


if __name__ == "__main__":
    raise SystemExit(run())
