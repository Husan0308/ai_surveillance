from __future__ import annotations

import os

from services.ml_service.app.config import CameraConfig
from .step1_independent_egl_v4 import V11Step1IndependentEglV4


class V11Step1Cam02LowLatV7(V11Step1IndependentEglV4):
    """V4 independent EGL display with one controlled decoder change.

    All cameras remain TCP, 100 ms RTSP latency, latest-only queue, GPU scaling
    and independent nveglglessink rendering. Only CAM-02 enables the internal
    nvv4l2decoder low-latency-mode. This is safe for the measured CAM-02 H.264
    IPPP stream because ffprobe observed zero B-frames.

    This DeepStream 7.1 build does not expose low-latency-mode on nvurisrcbin,
    so the property is applied to the decoder child via deep-element-added.
    """

    def __init__(self) -> None:
        raw = os.environ.get("V11_LOWLAT_CAMERAS", "CAM-02")
        self.lowlat_cameras = {item.strip() for item in raw.split(",") if item.strip()}
        self.decoder_lowlat_effective: dict[str, int] = {}
        super().__init__()

        known = {c.camera_id for c in self.cameras}
        unknown = sorted(self.lowlat_cameras.difference(known))
        if unknown:
            raise RuntimeError("V11 Step1 V7 unknown low-latency camera ids: " + ",".join(unknown))

        matrix = ",".join(
            f"{c.camera_id}:{int(c.camera_id in self.lowlat_cameras)}" for c in self.cameras
        )
        print(
            "CAMERA_V11_STEP1V7_ARCH base=v4-independent-egl decoder_ab=1 "
            "mux=0 tiler=0 detector=0 tracker=0 latest_only=1 transport=tcp",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP1V7_POLICY "
            f"latency_ms={self.latency_ms} drop_on_latency={int(self.drop_on_latency)} "
            f"low_latency_matrix={matrix} target=nvv4l2decoder",
            flush=True,
        )

    def _configure_rtsp_child(self, bin_obj, sub_bin, element, camera: CameraConfig) -> None:
        # Keep the proven V4 RTSP child configuration.
        super()._configure_rtsp_child(bin_obj, sub_bin, element, camera)

        factory = element.get_factory()
        factory_name = factory.get_name() if factory is not None else ""
        element_name = element.get_name() or ""
        if factory_name != "nvv4l2decoder" and "nvv4l2decoder" not in element_name:
            return

        cid = camera.camera_id
        requested = cid in self.lowlat_cameras
        prop = element.find_property("low-latency-mode")
        if prop is None:
            with self.lock:
                self.stats[cid].errors += 1
            print(
                "CAMERA_V11_STEP1V7_DECODER "
                f"camera={cid} low_latency=-1 property=missing element={element_name}",
                flush=True,
            )
            return

        element.set_property("low-latency-mode", bool(requested))
        effective = int(bool(element.get_property("low-latency-mode")))
        self.decoder_lowlat_effective[cid] = effective
        if effective != int(requested):
            with self.lock:
                self.stats[cid].errors += 1

        print(
            "CAMERA_V11_STEP1V7_DECODER "
            f"camera={cid} low_latency={effective} property=low-latency-mode "
            f"element={element_name} expected={int(requested)}",
            flush=True,
        )

    def _build_camera(self, index: int, camera: CameraConfig) -> None:
        cid = camera.camera_id
        requested_lowlat = int(cid in self.lowlat_cameras)
        safe = cid.lower().replace("-", "_")

        pipeline = self.Gst.Pipeline.new(f"v11_step1v7_{safe}")
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
        self._set_if(source, "select-rtp-protocol", 4)  # TCP-only for all cameras.
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
            "CAMERA_V11_STEP1V7_WINDOW "
            f"camera={cid} transport=tcp low_latency={requested_lowlat} "
            f"xid={self.wall.children[index]} overlay={overlay_ok} "
            f"tile={self.tile_width}x{self.tile_height}",
            flush=True,
        )


def main() -> int:
    return V11Step1Cam02LowLatV7().run()


if __name__ == "__main__":
    raise SystemExit(main())
