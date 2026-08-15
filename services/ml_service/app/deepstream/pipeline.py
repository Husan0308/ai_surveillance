from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from services.ml_service.app.config import Settings
from shared.frame_bus import LatestFrameWriter


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
    """Own the six-camera DeepStream graph and local latest-frame transport."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipeline: Gst.Pipeline | None = None
        self._loop: GLib.MainLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state = RuntimeState.STOPPED
        self._last_error: str | None = None

        self._frame_writers: dict[int, LatestFrameWriter] = {}
        self._frame_interval_ns = int(
            1_000_000_000 / self.settings.deepstream.frame_transport.max_fps
        )
        self._last_frame_accept_ns = [0] * len(self.settings.cameras)
        self._first_frame_seen = [False] * len(self.settings.cameras)

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

        self._close_frame_writers()

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
        frame = ds.frame_transport
        bool_text = lambda value: "true" if value else "false"

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
            output = " ".join(
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
            output = "mux. ! queue ! fakesink sync=false qos=false"

        source_parts: list[str] = []
        for index, camera in enumerate(self.settings.cameras):
            source = " ".join(
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
                    f"! tee name=source_tee_{index}",
                ]
            )

            mux_branch = " ".join(
                [
                    f"source_tee_{index}.",
                    "! queue",
                    "max-size-buffers=2",
                    "max-size-bytes=0",
                    "max-size-time=0",
                    "leaky=downstream",
                    f"! mux.sink_{index}",
                ]
            )

            source_parts.extend([source, mux_branch])

            if frame.enabled:
                preview_branch = " ".join(
                    [
                        f"source_tee_{index}.",
                        f"! queue name=frame_q_{index}",
                        "max-size-buffers=1",
                        "max-size-bytes=0",
                        "max-size-time=0",
                        "leaky=downstream",
                        f"! nvvideoconvert gpu-id={ds.gpu_id}",
                        f"! video/x-raw,format=RGBA,width={frame.width},height={frame.height},pixel-aspect-ratio=1/1",
                        f"! appsink name=frame_sink_{index}",
                        "emit-signals=true",
                        "sync=false",
                        "qos=false",
                        "max-buffers=1",
                        "drop=true",
                        "enable-last-sample=false",
                        "wait-on-eos=false",
                    ]
                )
                source_parts.append(preview_branch)

        return " ".join([mux, output, *source_parts])

    def _prepare_frame_writers(self) -> None:
        frame = self.settings.deepstream.frame_transport
        if not frame.enabled:
            return

        # RGBA is 4 bytes/pixel. Keep extra headroom for aligned strides.
        max_payload = frame.width * frame.height * 8

        for index, camera in enumerate(self.settings.cameras):
            writer = LatestFrameWriter(
                frame.directory,
                camera.camera_id,
                max_payload_bytes=max_payload,
            )
            self._frame_writers[index] = writer
            print(f"[FRAME_BUS] {camera.camera_id} -> {writer.path}", flush=True)

    def _close_frame_writers(self) -> None:
        writers = list(self._frame_writers.values())
        self._frame_writers.clear()

        for writer in writers:
            try:
                writer.close()
            except Exception:
                pass

    def _attach_frame_callbacks(self, pipeline: Gst.Pipeline) -> None:
        if not self.settings.deepstream.frame_transport.enabled:
            return

        for index, camera in enumerate(self.settings.cameras):
            queue = pipeline.get_by_name(f"frame_q_{index}")
            sink = pipeline.get_by_name(f"frame_sink_{index}")

            if queue is None or sink is None:
                raise RuntimeError(f"{camera.camera_id}: preview branch was not created")

            src_pad = queue.get_static_pad("src")
            if src_pad is None:
                raise RuntimeError(f"{camera.camera_id}: preview queue has no src pad")

            # Drop frames before the GPU color/scale conversion. This keeps preview
            # work bounded even when the source itself is 20-30 FPS.
            src_pad.add_probe(
                Gst.PadProbeType.BUFFER,
                self._frame_rate_probe,
                index,
            )
            sink.connect("new-sample", self._on_new_sample, index)

    def _frame_rate_probe(
        self,
        _pad: Gst.Pad,
        _info: Gst.PadProbeInfo,
        camera_index: int,
    ) -> Gst.PadProbeReturn:
        now = time.monotonic_ns()
        last = self._last_frame_accept_ns[camera_index]

        if last and now - last < self._frame_interval_ns:
            return Gst.PadProbeReturn.DROP

        self._last_frame_accept_ns[camera_index] = now
        return Gst.PadProbeReturn.OK

    def _on_new_sample(self, sink: Gst.Element, camera_index: int) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        if buffer is None or caps is None or caps.get_size() == 0:
            return Gst.FlowReturn.OK

        structure = caps.get_structure(0)
        try:
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))
        except (TypeError, ValueError):
            return Gst.FlowReturn.OK

        mapped, map_info = buffer.map(Gst.MapFlags.READ)
        if not mapped:
            camera_id = self.settings.cameras[camera_index].camera_id
            with self._lock:
                self._last_error = f"{camera_id}: appsink buffer map failed"
            return Gst.FlowReturn.OK

        try:
            if height <= 0:
                return Gst.FlowReturn.OK

            stride = max(width * 4, int(map_info.size // height))
            writer = self._frame_writers.get(camera_index)
            if writer is None:
                return Gst.FlowReturn.OK

            writer.publish(
                map_info.data,
                timestamp_ns=time.monotonic_ns(),
                width=width,
                height=height,
                stride=stride,
            )

            if not self._first_frame_seen[camera_index]:
                self._first_frame_seen[camera_index] = True
                camera_id = self.settings.cameras[camera_index].camera_id
                print(
                    f"[FRAME_BUS] {camera_id} first frame "
                    f"{width}x{height} stride={stride} bytes={map_info.size}",
                    flush=True,
                )
        except Exception as exc:
            camera_id = self.settings.cameras[camera_index].camera_id
            with self._lock:
                self._last_error = f"{camera_id}: frame transport error: {exc}"
        finally:
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK

    def _run(self) -> None:
        try:
            self._prepare_frame_writers()
            description = self.pipeline_description()
            pipeline = Gst.parse_launch(description)

            if not isinstance(pipeline, Gst.Pipeline):
                raise RuntimeError("GStreamer did not create a pipeline")

            self._pipeline = pipeline
            self._attach_frame_callbacks(pipeline)

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
            self._close_frame_writers()

    def _on_bus_message(self, _bus: Gst.Bus, message: Gst.Message) -> bool:
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            detail = str(err)
            if debug:
                detail = f"{detail} | {debug}"

            with self._lock:
                self._state = RuntimeState.ERROR
                self._last_error = detail

            print(f"[GSTREAMER ERROR] {detail}", flush=True)

        elif message.type == Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            detail = str(warning)
            if debug:
                detail = f"{detail} | {debug}"
            print(f"[GSTREAMER WARNING] {detail}", flush=True)

        elif message.type == Gst.MessageType.EOS:
            with self._lock:
                self._state = RuntimeState.ERROR
                self._last_error = "DeepStream pipeline reached EOS"
            print("[GSTREAMER] EOS", flush=True)

        return True
