from __future__ import annotations

import os
import threading
from typing import Mapping

from .person_tracking_final import CameraPersonTrackingFinal
from .qt_heatmap_bridge import QtHeatmapBridge


class CameraQtRuntime(CameraPersonTrackingFinal):
    """Real-time six-camera runtime for the native Qt monitoring UI.

    Source/decode/detection/tracking remains the working GPU pipeline. Only the
    presentation tail changes:

      NvDCF -> nvstreamdemux -> 6 x (queue -> RGBA -> nvdsosd -> nveglglessink)

    Each EGL sink is embedded into its own native Qt camera card via GstVideoOverlay.
    Movement heat is accumulated before demux and attached to each frame's own
    NvDsDisplayMeta, so every card gets only its own bbox + heatmap overlay.
    """

    def __init__(self) -> None:
        self.qt_heatmap = QtHeatmapBridge()
        self.qt_heatmap.configure(
            deposit=float(os.environ.get("CAMERA_V2_HEATMAP_DEPOSIT", "0.0025")),
            decay=float(os.environ.get("CAMERA_V2_HEATMAP_DECAY", "0.99992")),
            low=float(os.environ.get("CAMERA_V2_HEATMAP_LOW", "0.003")),
            yellow=float(os.environ.get("CAMERA_V2_HEATMAP_YELLOW", "0.070")),
            red=float(os.environ.get("CAMERA_V2_HEATMAP_RED", "0.180")),
            max_points=int(os.environ.get("CAMERA_V2_HEATMAP_MAX_POINTS", "18")),
        )
        self.qt_heatmap.reset()
        self._heatmap_visible = threading.Event()
        self.camera_sinks: dict[str, object] = {}
        self.camera_osds: dict[str, object] = {}
        self.demux_request_pads: list[object] = []
        self._window_handles: dict[str, int] = {}
        self._gst_video = None
        super().__init__()

    def set_heatmap_visible(self, visible: bool) -> None:
        if visible:
            self._heatmap_visible.set()
        else:
            self._heatmap_visible.clear()
        print(
            f"CAMERA_QT heatmap_visibility={'ON' if visible else 'OFF'} accumulation=ON",
            flush=True,
        )

    def heatmap_visible(self) -> bool:
        return self._heatmap_visible.is_set()

    def heatmap_updates_total(self) -> int:
        return self.qt_heatmap.updates_total()

    def heatmap_points_last(self) -> int:
        return self.qt_heatmap.points_last()

    def camera_person_count(self, camera_id: str) -> int:
        source_id = self.camera_index.get(camera_id)
        if source_id is None:
            return 0
        return self.qt_heatmap.current_count(source_id)

    def _remove_baseline_wall(self) -> None:
        # CameraWallV2 initially creates mux -> tiler -> wall_queue -> single sink.
        # The Qt layout needs six independent native windows, so detach only this
        # presentation tail. Source queues, mux and all detection side branches stay.
        try:
            self.mux.unlink(self.tiler)
        except Exception:
            pass
        try:
            self.tiler.unlink(self.wall_queue)
        except Exception:
            pass
        try:
            self.wall_queue.unlink(self.sink)
        except Exception:
            pass
        for element in (self.tiler, self.wall_queue, self.sink):
            try:
                element.set_state(self.Gst.State.NULL)
            except Exception:
                pass
            try:
                self.pipeline.remove(element)
            except Exception:
                pass

    def _configure_camera_sink(self, sink) -> None:
        self._set_if(sink, "sync", False)
        self._set_if(sink, "qos", False)
        self._set_if(sink, "async", False)
        self._set_if(sink, "enable-last-sample", False)
        self._set_if(sink, "max-lateness", -1)
        self._set_if(sink, "processing-deadline", 0)
        self._set_if(sink, "render-delay", 0)
        self._set_if(sink, "throttle-time", 0)
        self._set_if(sink, "force-aspect-ratio", True)
        self._set_if(sink, "gpu-id", self.gpu_id)

    def _request_demux_pad(self, index: int):
        name = f"src_{index}"
        request_simple = getattr(self.demux, "request_pad_simple", None)
        pad = request_simple(name) if request_simple else None
        if pad is None:
            try:
                pad = self.demux.get_request_pad(name)
            except Exception:
                pad = None
        if pad is None:
            # Some DeepStream builds expose these as sometimes-pads after linking.
            pad = self.demux.get_static_pad(name)
        if pad is None:
            raise RuntimeError(f"nvstreamdemux could not allocate {name}")
        self.demux_request_pads.append(pad)
        return pad

    def _install_osd_and_meta(self) -> None:
        # Called dynamically from CameraDetectionV2.__init__. At this point the
        # known-good six sources and nvstreammux already exist.
        for plugin in ("nvtracker", "nvstreamdemux", "nvvideoconvert", "nvdsosd", "nveglglessink"):
            if self.Gst.ElementFactory.find(plugin) is None:
                raise RuntimeError(f"Qt runtime missing DeepStream plugin: {plugin}")

        self._remove_baseline_wall()

        tracker = self._make("nvtracker", "person_nvdcf_tracker")
        self._set_if(tracker, "tracker-width", self.tracker_width)
        self._set_if(tracker, "tracker-height", self.tracker_height)
        tracker.set_property("ll-lib-file", str(self.tracker_lib))
        tracker.set_property("ll-config-file", str(self.tracker_config))
        self._set_if(tracker, "gpu-id", self.gpu_id)
        self._set_if(tracker, "compute-hw", 1)
        self._set_if(tracker, "enable-batch-process", True)
        self._set_if(tracker, "display-tracking-id", False)
        self._set_if(tracker, "tracking-id-reset-mode", 1)
        self._set_if(tracker, "user-meta-pool-size", 64)

        self.demux = self._make("nvstreamdemux", "camera_qt_demux")
        self.pipeline.add(tracker)
        self.pipeline.add(self.demux)
        if not self.mux.link(tracker):
            raise RuntimeError("failed nvstreammux -> nvtracker for Qt runtime")
        if not tracker.link(self.demux):
            raise RuntimeError("failed nvtracker -> nvstreamdemux for Qt runtime")

        for camera in self.cameras:
            cid = camera.camera_id
            index = self.camera_index[cid]
            q = self._make("queue", f"qt_display_queue_{index}")
            convert = self._make("nvvideoconvert", f"qt_display_convert_{index}")
            caps = self._make("capsfilter", f"qt_display_caps_{index}")
            osd = self._make("nvdsosd", f"qt_display_osd_{index}")
            sink = self._make("nveglglessink", f"qt_display_sink_{index}")

            self._set_if(q, "max-size-buffers", 1)
            self._set_if(q, "max-size-bytes", 0)
            self._set_if(q, "max-size-time", 0)
            self._set_if(q, "leaky", 2)
            self._set_if(q, "silent", True)
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
            self._configure_camera_sink(sink)

            for element in (q, convert, caps, osd, sink):
                self.pipeline.add(element)
            if not q.link(convert) or not convert.link(caps) or not caps.link(osd) or not osd.link(sink):
                raise RuntimeError(f"{cid}: failed Qt display branch link")

            demux_pad = self._request_demux_pad(index)
            if demux_pad.link(q.get_static_pad("sink")) != self.Gst.PadLinkReturn.OK:
                raise RuntimeError(f"{cid}: nvstreamdemux -> Qt display queue failed")

            self.camera_sinks[cid] = sink
            self.camera_osds[cid] = osd

        # One fresh detector result is injected at mux output. NvDCF then tracks
        # every live frame before demux. Our overridden tracker probe performs
        # movement-heat update + bbox style on the same current frame.
        self.mux.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER, self._inject_detector_probe
        )
        tracker.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER, self._tracker_probe
        )
        tracker.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER, self._wall_probe
        )
        self.tracker = tracker
        # Compatibility: no single tiled OSD exists in Qt mode.
        self.osd = None

    def _tracker_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is not None:
            # First: tight raw NvDCF bbox -> movement-only floor heat. It also
            # attaches per-frame display-meta circles when visibility is ON.
            self.qt_heatmap.process(buffer, self._heatmap_visible.is_set())
            # Then style/smooth ONLY the real current-frame person boxes.
            count = self.bridge.style_and_count_tracked(buffer)
            if count >= 0:
                with self.det_lock:
                    self.tracked_now = count
                    self.tracker_frames += 1
        return self.Gst.PadProbeReturn.OK

    def _sink_stats(self) -> tuple[int | None, int | None]:
        rendered_total = 0
        dropped_total = 0
        have = False
        for sink in self.camera_sinks.values():
            if sink.find_property("stats") is None:
                continue
            try:
                stats = sink.get_property("stats")
                if stats.has_field("rendered"):
                    rendered_total += int(stats.get_value("rendered"))
                    have = True
                if stats.has_field("dropped"):
                    dropped_total += int(stats.get_value("dropped"))
                    have = True
            except Exception:
                pass
        return (rendered_total, dropped_total) if have else (None, None)

    def bind_window_handles(self, handles: Mapping[str, int]) -> None:
        """Bind six already-created Qt native WIds before the pipeline starts.

        We set the handles immediately AND install the official synchronous
        prepare-window-handle handler. The latter is required because a video sink
        may request its target window from a streaming thread at state-change time.
        """
        import gi
        gi.require_version("GstVideo", "1.0")
        from gi.repository import GstVideo

        self._gst_video = GstVideo
        self._window_handles.clear()
        for cid, sink in self.camera_sinks.items():
            handle = int(handles.get(cid, 0))
            if handle <= 0:
                raise RuntimeError(f"{cid}: Qt native window handle is invalid")
            self._window_handles[sink.get_name()] = handle
            GstVideo.VideoOverlay.set_window_handle(sink, handle)
            try:
                GstVideo.VideoOverlay.handle_events(sink, False)
            except Exception:
                pass

        self.bus.set_sync_handler(self._bus_sync_handler, None)
        print(
            "CAMERA_QT overlay_handles_bound="
            + ",".join(f"{cid}:{int(handles[cid])}" for cid in self.camera_sinks),
            flush=True,
        )

    def _bus_sync_handler(self, _bus, message, _user_data):
        GstVideo = self._gst_video
        if GstVideo is None:
            return self.Gst.BusSyncReply.PASS
        try:
            if not GstVideo.is_video_overlay_prepare_window_handle_message(message):
                return self.Gst.BusSyncReply.PASS
            src = message.src
            name = src.get_name() if src is not None else ""
            handle = self._window_handles.get(name, 0)
            if handle:
                GstVideo.VideoOverlay.set_window_handle(src, handle)
                try:
                    GstVideo.VideoOverlay.handle_events(src, False)
                except Exception:
                    pass
                return self.Gst.BusSyncReply.DROP
        except Exception as exc:
            print(f"CAMERA_QT prepare-window-handle warning: {exc}", flush=True)
        return self.Gst.BusSyncReply.PASS

    def update_render_rectangle(self, camera_id: str, width: int, height: int) -> None:
        if self._gst_video is None:
            return
        sink = self.camera_sinks.get(camera_id)
        if sink is None or width < 2 or height < 2:
            return
        try:
            self._gst_video.VideoOverlay.set_render_rectangle(sink, 0, 0, int(width), int(height))
            self._gst_video.VideoOverlay.expose(sink)
        except Exception:
            pass

    def release_qt_pads(self) -> None:
        for pad in self.demux_request_pads:
            try:
                self.demux.release_request_pad(pad)
            except Exception:
                pass
        self.demux_request_pads.clear()
