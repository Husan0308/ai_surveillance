from __future__ import annotations

from pathlib import Path

import gi


gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo  # noqa: E402

Gst.init(None)


def _gst_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class NativeShmRenderer:
    """Render one ML native-resolution SHM stream into an X11 Qt window.

    ML sends system-memory NV12 at the decoder's original resolution. The
    frontend converts it back to NVMM/RGBA for nveglglessink. There is no JPEG,
    QImage or QPixmap path on the normal renderer.
    """

    def __init__(
        self,
        camera_id: str,
        window_id: int,
        socket_path: str | Path,
        width: int,
        height: int,
        fps: int,
        gpu_id: int = 0,
        pixel_format: str = "NV12",
    ) -> None:
        self.camera_id = camera_id
        self.window_id = int(window_id)
        self.socket_path = Path(socket_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(round(float(fps))))
        self.gpu_id = int(gpu_id)
        self.pixel_format = str(pixel_format or "NV12").upper()
        self.pipeline = None
        self.bus = None
        self.sink = None
        self.state = "stopped"
        self.last_error = ""
        self.last_warning = ""

    @staticmethod
    def required_plugins() -> tuple[str, ...]:
        return ("shmsrc", "nvvideoconvert", "nveglglessink")

    @classmethod
    def missing_plugins(cls) -> tuple[str, ...]:
        return tuple(
            name for name in cls.required_plugins() if Gst.ElementFactory.find(name) is None
        )

    def _pipeline_text(self) -> str:
        return " ".join(
            [
                "shmsrc",
                f"socket-path={_gst_quote(str(self.socket_path))}",
                "is-live=true",
                "do-timestamp=true",
                "!",
                (
                    f"video/x-raw,format={self.pixel_format},width={self.width},"
                    f"height={self.height},framerate={self.fps}/1"
                ),
                "!",
                "queue",
                "max-size-buffers=1",
                "max-size-bytes=0",
                "max-size-time=0",
                "leaky=downstream",
                "silent=true",
                "!",
                "nvvideoconvert",
                f"gpu-id={self.gpu_id}",
                "!",
                "video/x-raw(memory:NVMM),format=RGBA",
                "!",
                "nveglglessink",
                "name=video_sink",
                "sync=false",
                "qos=false",
                "async=false",
            ]
        )

    def _bind(self, overlay=None) -> None:
        target = overlay if overlay is not None else self.sink
        if target is None or self.window_id <= 0:
            return
        GstVideo.VideoOverlay.set_window_handle(target, self.window_id)
        try:
            GstVideo.VideoOverlay.handle_events(target, False)
        except Exception:
            pass

    def _sync_handler(self, _bus, message, _data=None):
        try:
            prepare = GstVideo.is_video_overlay_prepare_window_handle_message(message)
        except Exception:
            structure = message.get_structure()
            prepare = bool(structure and structure.get_name() == "prepare-window-handle")
        if not prepare:
            return Gst.BusSyncReply.PASS
        try:
            self._bind(message.src)
            return Gst.BusSyncReply.DROP
        except Exception as exc:
            self.last_error = f"video overlay bind failed: {exc}"
            self.state = "error"
            return Gst.BusSyncReply.PASS

    def start(self) -> None:
        if self.pipeline is not None:
            return
        missing = self.missing_plugins()
        if missing:
            raise RuntimeError(f"missing GStreamer plugins: {','.join(missing)}")
        if self.window_id <= 0:
            raise RuntimeError("invalid Qt native window id")
        if self.width <= 0 or self.height <= 0:
            raise RuntimeError(f"invalid native stream size {self.width}x{self.height}")
        if not self.socket_path.exists():
            raise RuntimeError(f"shared-memory socket not ready: {self.socket_path}")

        self.pipeline = Gst.parse_launch(self._pipeline_text())
        self.bus = self.pipeline.get_bus()
        self.sink = self.pipeline.get_by_name("video_sink")
        if self.sink is None:
            self.stop()
            raise RuntimeError("nveglglessink was not created")
        self.bus.set_sync_handler(self._sync_handler, None)
        self._bind()
        if self.sink.find_property("force-aspect-ratio") is not None:
            self.sink.set_property("force-aspect-ratio", True)

        result = self.pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            self.poll()
            error = self.last_error or "native GStreamer renderer failed to enter PLAYING"
            self.stop()
            raise RuntimeError(error)
        self.state = "starting"

    def poll(self) -> str:
        bus = self.bus
        if bus is None:
            return self.state
        while True:
            message = bus.pop_filtered(
                Gst.MessageType.ERROR
                | Gst.MessageType.WARNING
                | Gst.MessageType.EOS
                | Gst.MessageType.STATE_CHANGED
            )
            if message is None:
                break
            source = message.src.get_name() if message.src is not None else "unknown"
            if message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                self.last_error = f"{source}: {err.message} | {debug or ''}"
                self.state = "error"
            elif message.type == Gst.MessageType.WARNING:
                err, debug = message.parse_warning()
                self.last_warning = f"{source}: {err.message} | {debug or ''}"
            elif message.type == Gst.MessageType.EOS:
                self.last_error = f"EOS from {source}"
                self.state = "error"
            elif message.type == Gst.MessageType.STATE_CHANGED and message.src == self.pipeline:
                _old, new, _pending = message.parse_state_changed()
                if new == Gst.State.PLAYING:
                    self.state = "playing"
        return self.state

    def expose(self) -> None:
        if self.sink is None:
            return
        try:
            GstVideo.VideoOverlay.expose(self.sink)
        except Exception:
            pass

    def stop(self) -> None:
        pipeline = self.pipeline
        self.pipeline = None
        self.bus = None
        self.sink = None
        if pipeline is not None:
            try:
                pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        if self.state != "error":
            self.state = "stopped"
