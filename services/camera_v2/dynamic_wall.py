from __future__ import annotations

import math
import os

from services.ml_service.app.config import load_settings

from .main import CameraWallV2, SourceRuntime


class DynamicCameraWallV2(CameraWallV2):
    """CameraWallV2 initialization without the historical exactly-six restriction.

    The hot path stays identical: nvurisrcbin/NVDEC -> queue(1) -> nvstreammux ->
    nvmultistreamtiler -> queue(1) -> nveglglessink. Only batch/grid dimensions are
    derived from the enabled camera inventory so Settings can add or disable RTSP
    cameras without a second video architecture.
    """

    def __init__(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst

        Gst.init(None)
        self.Gst = Gst
        self.GLib = GLib
        self.settings = load_settings()
        self.cameras = list(self.settings.cameras)
        if not 1 <= len(self.cameras) <= 16:
            raise RuntimeError(
                f"Camera V2 supports 1..16 enabled cameras, found {len(self.cameras)}"
            )

        ds = self.settings.deepstream
        self.gpu_id = int(os.environ.get("CAMERA_V2_GPU_ID", ds.gpu_id))
        self.rtsp_latency_ms = max(
            40, int(os.environ.get("CAMERA_V2_RTSP_LATENCY_MS", "100"))
        )
        self.udp_buffer_size = max(
            1_048_576,
            int(os.environ.get("CAMERA_V2_UDP_BUFFER_SIZE", str(8 * 1024 * 1024))),
        )
        self.extra_surfaces = max(
            2, min(16, int(os.environ.get("CAMERA_V2_EXTRA_SURFACES", "6")))
        )
        self.low_latency_mode = os.environ.get(
            "CAMERA_V2_LOW_LATENCY_MODE", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}

        self.source_fps = max(
            1, int(os.environ.get("CAMERA_V2_SOURCE_FPS", "20"))
        )
        self.mux_timeout_us = max(
            5_000,
            int(
                os.environ.get(
                    "CAMERA_V2_MUX_TIMEOUT_US",
                    str(round(1_000_000 / self.source_fps)),
                )
            ),
        )
        self.frame_width = max(
            320, int(os.environ.get("CAMERA_V2_FRAME_WIDTH", "1280"))
        )
        self.frame_height = max(
            180, int(os.environ.get("CAMERA_V2_FRAME_HEIGHT", "720"))
        )

        default_columns = min(3, max(1, len(self.cameras)))
        self.tiler_columns = max(
            1,
            min(
                len(self.cameras),
                int(os.environ.get("CAMERA_V2_TILER_COLUMNS", str(default_columns))),
            ),
        )
        self.tiler_rows = max(1, math.ceil(len(self.cameras) / self.tiler_columns))
        default_wall_width = 640 * self.tiler_columns
        default_wall_height = 360 * self.tiler_rows
        self.wall_width = max(
            640,
            int(os.environ.get("CAMERA_V2_WALL_WIDTH", str(default_wall_width))),
        )
        self.wall_height = max(
            360,
            int(os.environ.get("CAMERA_V2_WALL_HEIGHT", str(default_wall_height))),
        )

        self.stats = {cam.camera_id: SourceRuntime() for cam in self.cameras}
        self.queues: dict[str, object] = {}
        self.sources: dict[str, object] = {}
        self._request_pads: list[object] = []
        self._warning_last: dict[str, float] = {}
        self._stopping = False

        self._preflight()
        self.pipeline = Gst.Pipeline.new("camera-v2-gpu-wall")
        if self.pipeline is None:
            raise RuntimeError("Could not create GStreamer pipeline")

        self.mux = self._make("nvstreammux", "camera_v2_mux")
        self.tiler = self._make("nvmultistreamtiler", "camera_v2_tiler")
        self.wall_queue = self._make("queue", "camera_v2_wall_queue")
        self.sink = self._make("nveglglessink", "camera_v2_sink")

        self._configure_mux()
        self._configure_tiler()
        self._configure_wall_queue()
        self._configure_sink()

        for element in (self.mux, self.tiler, self.wall_queue, self.sink):
            self.pipeline.add(element)

        self._require_link(self.mux, self.tiler, "nvstreammux -> nvmultistreamtiler")
        self._require_link(self.tiler, self.wall_queue, "nvmultistreamtiler -> wall queue")
        self._require_link(self.wall_queue, self.sink, "wall queue -> nveglglessink")

        for index, camera in enumerate(self.cameras):
            self._add_camera(index, camera)

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)
        self.loop = GLib.MainLoop()
        GLib.timeout_add_seconds(5, self._print_stats)

    def _configure_mux(self) -> None:
        self._set_if(self.mux, "batch-size", len(self.cameras))
        self._set_if(self.mux, "live-source", True)
        self._set_if(self.mux, "width", self.frame_width)
        self._set_if(self.mux, "height", self.frame_height)
        self._set_if(self.mux, "enable-padding", False)
        self._set_if(self.mux, "batched-push-timeout", self.mux_timeout_us)
        self._set_if(self.mux, "sync-inputs", False)
        self._set_if(self.mux, "max-latency", 0)
        self._set_if(self.mux, "buffer-pool-size", max(8, len(self.cameras) + 2))
        self._set_if(self.mux, "nvbuf-memory-type", 2)
        self._set_if(self.mux, "gpu-id", self.gpu_id)

    def _configure_tiler(self) -> None:
        self._set_if(self.tiler, "rows", self.tiler_rows)
        self._set_if(self.tiler, "columns", self.tiler_columns)
        self._set_if(self.tiler, "width", self.wall_width)
        self._set_if(self.tiler, "height", self.wall_height)
        self._set_if(self.tiler, "gpu-id", self.gpu_id)
        self._set_if(self.tiler, "nvbuf-memory-type", 2)
