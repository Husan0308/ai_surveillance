from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from services.ml_service.app.config import Settings


class RuntimeState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    PLAYING = "playing"
    ERROR = "error"


@dataclass
class RuntimeSnapshot:
    state: RuntimeState
    camera_count: int
    last_error: str | None


class DeepStreamRuntime:
    """Owns the complete six-camera DeepStream graph.

    Phase 1 intentionally contains no detector, tracker, ReID, face recognition,
    database, recording, or analytics elements.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipeline: Gst.Pipeline | None = None
        self._loop: GLib.MainLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state = RuntimeState.STOPPED
        self._last_error: str | None = None

        Gst.init(None)

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._state = RuntimeState.STARTING
            self._last_error = None
            self._thread = threading.Thread(
                target=self._run,
                name="deepstream-runtime",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        loop = self._loop
        pipeline = self._pipeline

        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
        if loop is not None and loop.is_running():
            loop.quit()

        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5)

        with self._lock:
            self._pipeline = None
            self._loop = None
            self._thread = None
            self._state = RuntimeState.STOPPED

    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            return RuntimeSnapshot(
                state=self._state,
                camera_count=len(self.settings.cameras),
                last_error=self._last_error,
            )

    def pipeline_description(self) -> str:
        ds = self.settings.deepstream
        bool_text = lambda value: "true" if value else "false"

        source_parts: list[str] = []
        for index, camera in enumerate(self.settings.cameras):
            source_parts.append(
                " ".join(
                    [
                        "nvurisrcbin",
                        f"name=src_{index}",
                        f'uri="{camera.uri}"',
                        f"gpu-id={ds.gpu_id}",
                        f"latency={ds.latency_ms}",
                        f"drop-on-latency={bool_text(ds.drop_on_latency)}",
                        f"rtsp-reconnect-interval={ds.reconnect_interval_sec}",
                        f"rtsp-reconnect-attempts={ds.reconnect_attempts}",
                        f"num-extra-surfaces={ds.decoder_extra_surfaces}",
                        f"cudadec-memtype={ds.cudadec_memtype}",
                        f"udp-buffer-size={ds.udp_buffer_size}",
                        f"select-rtp-protocol={ds.rtp_protocol}",
                        "disable-audio=true",
                        "message-forward=true",
                        "! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream",
                        f"! mux.sink_{index}",
                    ]
                )
            )

        mux = " ".join(
            [
                "nvstreammux",
                "name=mux",
                f"batch-size={ds.batch_size}",
                f"width={ds.mux_width}",
                f"height={ds.mux_height}",
                f"live-source={bool_text(ds.live_source)}",
                f"batched-push-timeout={ds.batched_push_timeout_us}",
                f"sync-inputs={bool_text(ds.sync_inputs)}",
                f"gpu-id={ds.gpu_id}",
            ]
        )

        if ds.display_enabled:
            sink = " ".join(
                [
                    "mux.",
                    "! queue",
                    "! nvmultistreamtiler",
                    f"rows={ds.display_rows}",
                    f"columns={ds.display_columns}",
                    f"width={ds.display_width}",
                    f"height={ds.display_height}",
                    f"gpu-id={ds.gpu_id}",
                    "! queue",
                    "! nveglglessink sync=false qos=false",
                ]
            )
        else:
            sink = "mux. ! queue ! fakesink sync=false"

        return " ".join([mux, sink, *source_parts])

    def _run(self) -> None:
        try:
            description = self.pipeline_description()
            pipeline = Gst.parse_launch(description)
            if not isinstance(pipeline, Gst.Pipeline):
                raise RuntimeError("GStreamer did not create a pipeline")

            self._pipeline = pipeline
            bus = pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)

            loop = GLib.MainLoop()
            self._loop = loop

            result = pipeline.set_state(Gst.State.PLAYING)
            if result == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("DeepStream pipeline failed to enter PLAYING")

            with self._lock:
                self._state = RuntimeState.PLAYING

            loop.run()
        except Exception as exc:
            with self._lock:
                self._state = RuntimeState.ERROR
                self._last_error = str(exc)
        finally:
            pipeline = self._pipeline
            if pipeline is not None:
                pipeline.set_state(Gst.State.NULL)

    def _on_bus_message(self, _bus: Gst.Bus, message: Gst.Message) -> bool:
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            detail = str(err)
            if debug:
                detail = f"{detail} | {debug}"
            with self._lock:
                self._state = RuntimeState.ERROR
                self._last_error = detail

        elif message.type == Gst.MessageType.EOS:
            with self._lock:
                self._state = RuntimeState.ERROR
                self._last_error = "DeepStream pipeline reached EOS"

        return True
