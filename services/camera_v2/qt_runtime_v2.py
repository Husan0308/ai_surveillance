from __future__ import annotations

from typing import Mapping

from .qt_runtime import CameraQtRuntime


class CameraQtRuntimeV2(CameraQtRuntime):
    """Qt-safe embedding layer for CameraQtRuntime.

    The DeepStream graph stays unchanged through detection/tracking. This class
    fixes the GstVideoOverlay lifecycle and keeps the six post-demux display
    branches lightweight for the GTX 1050 Ti by rendering each card at 768x432.
    """

    def __init__(self) -> None:
        self._sink_to_camera: dict[str, str] = {}
        self._prepared_sinks: set[str] = set()
        self._render_rectangles: dict[str, tuple[int, int]] = {}
        super().__init__()

        # Tracking still runs on the mux/tracker resolution. Only the final Qt card
        # branches are scaled down before RGBA/OSD, which saves display memory and
        # OSD work without reducing detector/tracker quality.
        display_w = 768
        display_h = 432
        for cid, index in self.camera_index.items():
            capsfilter = self.pipeline.get_by_name(f"qt_display_caps_{index}")
            if capsfilter is None:
                raise RuntimeError(f"{cid}: Qt display capsfilter not found")
            capsfilter.set_property(
                "caps",
                self.Gst.Caps.from_string(
                    f"video/x-raw(memory:NVMM),format=RGBA,width={display_w},height={display_h},pixel-aspect-ratio=1/1"
                ),
            )
        print(f"CAMERA_QT_V2 display_branches={display_w}x{display_h}x6", flush=True)

    def bind_window_handles(self, handles: Mapping[str, int]) -> None:
        import gi

        gi.require_version("GstVideo", "1.0")
        from gi.repository import GstVideo

        self._gst_video = GstVideo
        self._window_handles.clear()
        self._sink_to_camera.clear()
        self._prepared_sinks.clear()

        for cid, sink in self.camera_sinks.items():
            handle = int(handles.get(cid, 0))
            if handle <= 0:
                raise RuntimeError(f"{cid}: Qt native window handle is invalid")
            name = sink.get_name()
            self._window_handles[name] = handle
            self._sink_to_camera[name] = cid

        # GStreamer requires prepare-window-handle to be handled synchronously.
        # We cache integer WIds on the Qt thread, then the streaming callback only
        # forwards those integers to GstVideoOverlay (no Qt calls from that thread).
        self.bus.set_sync_handler(self._bus_sync_handler_v2, None)
        print(
            "CAMERA_QT_V2 handles_cached="
            + ",".join(f"{cid}:{int(handles[cid])}" for cid in self.camera_sinks),
            flush=True,
        )

    def _bus_sync_handler_v2(self, _bus, message, _user_data):
        GstVideo = self._gst_video
        if GstVideo is None:
            return self.Gst.BusSyncReply.PASS
        try:
            if not GstVideo.is_video_overlay_prepare_window_handle_message(message):
                return self.Gst.BusSyncReply.PASS
            src = message.src
            if src is None:
                return self.Gst.BusSyncReply.PASS
            sink_name = src.get_name()
            handle = int(self._window_handles.get(sink_name, 0))
            if handle <= 0:
                print(f"CAMERA_QT_V2 missing handle for {sink_name}", flush=True)
                return self.Gst.BusSyncReply.PASS

            GstVideo.VideoOverlay.set_window_handle(src, handle)
            try:
                GstVideo.VideoOverlay.handle_events(src, False)
            except Exception:
                pass

            self._prepared_sinks.add(sink_name)
            cid = self._sink_to_camera.get(sink_name)
            if cid:
                rect = self._render_rectangles.get(cid)
                if rect:
                    width, height = rect
                    try:
                        GstVideo.VideoOverlay.set_render_rectangle(src, 0, 0, width, height)
                    except Exception:
                        pass
            print(f"CAMERA_QT_V2 prepared {sink_name} handle={handle}", flush=True)
            return self.Gst.BusSyncReply.DROP
        except Exception as exc:
            print(f"CAMERA_QT_V2 prepare-window-handle warning: {exc}", flush=True)
            return self.Gst.BusSyncReply.PASS

    def update_render_rectangle(self, camera_id: str, width: int, height: int) -> None:
        width = int(width)
        height = int(height)
        if width < 2 or height < 2:
            return
        self._render_rectangles[camera_id] = (width, height)

        GstVideo = self._gst_video
        sink = self.camera_sinks.get(camera_id)
        if GstVideo is None or sink is None:
            return
        if sink.get_name() not in self._prepared_sinks:
            return
        try:
            GstVideo.VideoOverlay.set_render_rectangle(sink, 0, 0, width, height)
            GstVideo.VideoOverlay.expose(sink)
        except Exception:
            pass
