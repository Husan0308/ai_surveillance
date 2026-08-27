from __future__ import annotations

import os

from services.ml_service.app.config import CameraConfig
from .step1_independent_egl_v4 import V11Step1IndependentEglV4


class V11Step1Cam02UdpV5(V11Step1IndependentEglV4):
    """V4 independent EGL display with one controlled transport change.

    CAM-02 uses UDP-only at the underlying rtspsrc. All other cameras remain
    TCP-only. Display, decoder, latest-only queues, GPU scaling, sink policy and
    100 ms RTSP latency are unchanged from the V4 DS100 comparison run.
    """

    def __init__(self) -> None:
        raw = os.environ.get("V11_UDP_CAMERAS", "CAM-02")
        self.udp_cameras = {item.strip() for item in raw.split(",") if item.strip()}
        super().__init__()
        unknown = sorted(self.udp_cameras.difference({c.camera_id for c in self.cameras}))
        if unknown:
            raise RuntimeError("V11 Step1 V5 unknown UDP camera ids: " + ",".join(unknown))
        matrix = ",".join(
            f"{c.camera_id}:{self._transport_for(c.camera_id)}" for c in self.cameras
        )
        print(
            "CAMERA_V11_STEP1V5_ARCH base=v4-independent-egl transport_ab=1 "
            "mux=0 tiler=0 detector=0 tracker=0 latest_only=1",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP1V5_POLICY "
            f"latency_ms={self.latency_ms} drop_on_latency={int(self.drop_on_latency)} "
            f"udp_buffer_size={self.udp_buffer_size} transports={matrix}",
            flush=True,
        )

    def _transport_for(self, cid: str) -> str:
        return "udp" if cid in self.udp_cameras else "tcp"

    def _configure_rtsp_child(self, _bin, _sub_bin, element, camera: CameraConfig) -> None:
        factory = element.get_factory()
        if factory is None or factory.get_name() != "rtspsrc":
            return

        transport = self._transport_for(camera.camera_id)
        if camera.username:
            self._set_if(element, "user-id", camera.username)
            self._set_if(element, "user-pw", camera.password)

        # GstRTSPLowerTrans flags: UDP=1, TCP=4. Restrict only CAM-02 to UDP.
        self._set_if(element, "protocols", 1 if transport == "udp" else 4)
        self._set_if(element, "latency", self.latency_ms)
        self._set_if(element, "drop-on-latency", self.drop_on_latency)
        self._set_if(element, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(element, "do-rtsp-keep-alive", True)
        print(
            "CAMERA_V11_STEP1V5_RTSP "
            f"camera={camera.camera_id} transport={transport} latency_ms={self.latency_ms} "
            f"drop_on_latency={int(self.drop_on_latency)}",
            flush=True,
        )

    def _build_camera(self, index: int, camera: CameraConfig) -> None:
        cid = camera.camera_id
        transport = self._transport_for(cid)
        safe = cid.lower().replace("-", "_")
        pipeline = self.Gst.Pipeline.new(f"v11_step1v5_{safe}")
        if pipeline is None:
            raise RuntimeError(f"{cid}: could not create pipeline")

        source = self._make("nvurisrcbin", f"source_{safe}")
        queue = self._make("queue", f"latest_{safe}")
        convert = self._make("nvvideoconvert", f"scale_{safe}")
        capsfilter = self._make("capsfilter", f"caps_{safe}")
        sink = self._make("nveglglessink", f"sink_{safe}")

        self._configure_latest_queue(queue)
        source.connect("deep-element-added", self._configure_rtsp_child, camera)
        source.connect("pad-added", self._on_source_pad_added, cid)
        source.set_property("uri", camera.uri)
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "gpu-id", self.gpu_id)
        self._set_if(source, "latency", self.latency_ms)
        self._set_if(source, "drop-on-latency", self.drop_on_latency)
        self._set_if(source, "num-extra-surfaces", self.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "udp-buffer-size", self.udp_buffer_size)
        # nvurisrcbin exposes 4=TCP-only and 0=multi. For UDP-only we allow
        # multi here and force UDP on the child rtspsrc protocols property.
        self._set_if(source, "select-rtp-protocol", 4 if transport == "tcp" else 0)
        self._set_if(source, "rtsp-reconnect-interval", self.reconnect_sec)
        self._set_if(source, "rtsp-reconnect-attempts", -1)
        self._set_if(source, "message-forward", True)
        self._set_if(source, "async-handling", True)

        self._set_if(convert, "gpu-id", self.gpu_id)
        self._set_if(convert, "nvbuf-memory-type", 0)
        self._set_if(convert, "interpolation-method", self.interpolation)
        capsfilter.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM),format=NV12,width={self.tile_width},height={self.tile_height}"
            ),
        )

        self._set_if(sink, "sync", False)
        self._set_if(sink, "qos", False)
        self._set_if(sink, "async", False)
        self._set_if(sink, "enable-last-sample", False)
        self._set_if(sink, "force-aspect-ratio", False)

        for element in (source, queue, convert, capsfilter, sink):
            pipeline.add(element)
        self._require_link(queue, convert, f"{cid}:queue->convert")
        self._require_link(convert, capsfilter, f"{cid}:convert->caps")
        self._require_link(capsfilter, sink, f"{cid}:caps->sink")

        sink_pad = sink.get_static_pad("sink")
        if sink_pad is None:
            raise RuntimeError(f"{cid}: sink pad missing")
        sink_pad.add_probe(self.Gst.PadProbeType.BUFFER, self._render_probe, cid)

        try:
            self.GstVideo.VideoOverlay.set_window_handle(sink, int(self.wall.children[index]))
            overlay_ok = 1
        except Exception as exc:
            raise RuntimeError(f"{cid}: GstVideoOverlay window binding failed: {exc}") from exc

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, cid)

        self.pipelines[cid] = pipeline
        self.sources[cid] = source
        self.queues[cid] = queue
        self.converters[cid] = convert
        self.capsfilters[cid] = capsfilter
        self.sinks[cid] = sink

        print(
            "CAMERA_V11_STEP1V5_WINDOW "
            f"camera={cid} transport={transport} xid={self.wall.children[index]} "
            f"overlay={overlay_ok} tile={self.tile_width}x{self.tile_height}",
            flush=True,
        )


def main() -> int:
    return V11Step1Cam02UdpV5().run()


if __name__ == "__main__":
    raise SystemExit(main())
