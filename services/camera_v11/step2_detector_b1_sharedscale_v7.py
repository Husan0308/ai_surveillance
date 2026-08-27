from __future__ import annotations

from collections import deque

from services.ml_service.app.config import CameraConfig
from .step2_detector_b1_v6 import V11Step2DetectorB1V6
from .step2_detector_only import DETECT_CONTENT_H, DETECT_W


class V11Step2DetectorB1SharedScaleV7(V11Step2DetectorB1V6):
    """Step2 V7: reuse the frozen display scale before detector CPU export.

    Old V5/V6 topology:
        NVDEC/full-res -> tee
          -> display scale 640x360 NVMM
          -> detector scale full-res->672x378 + NVMM->RAW BGRx

    V7 topology:
        NVDEC/full-res -> latest1 -> ONE shared scale 640x360 NVMM -> tee
          -> latest1 -> frozen EGL display
          -> latest1/demand -> 640x360->672x378 + RAW BGRx -> appsink

    This keeps the visible Step1 geometry exactly 640x360 while removing the
    expensive second full-resolution resize on detector requests. Detector duty
    stays fixed at 2 Hz/camera for a clean conversion-path A/B test.
    """

    def __init__(self) -> None:
        self.display_output_queues: dict[str, object] = {}
        super().__init__()
        print(
            "CAMERA_V11_STEP2V7_ARCH "
            "base=step2-v6 detector=trt86-batch1 per_camera=2Hz "
            "scale_once_before_tee=1 shared_display_surface=1 fullres_detector_resize=0 "
            "tracker=0 osd=0 reid=0 face=0",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP2V7_POLICY "
            "display_scale=640x360/NVMM detector_source=shared-640x360/NVMM "
            "detector_output=672x378/BGRx-RAW batch=1 prefetch=0 latest_only=1",
            flush=True,
        )

    def _build_camera(self, index: int, camera: CameraConfig) -> None:
        cid = camera.camera_id
        safe = cid.lower().replace("-", "_")
        requested_lowlat = int(cid in self.lowlat_cameras)

        pipeline = self.Gst.Pipeline.new(f"v11_step2v7_{safe}")
        if pipeline is None:
            raise RuntimeError(f"{cid}: could not create Step2 V7 pipeline")

        source = self._make("nvurisrcbin", f"source_{safe}")
        source_q = self._make("queue", f"latest_{safe}")
        shared_convert = self._make("nvvideoconvert", f"scale_{safe}")
        shared_caps = self._make("capsfilter", f"caps_{safe}")
        tee = self._make("tee", f"shared_tee_{safe}")

        display_q = self._make("queue", f"display_after_scale_{safe}")
        display_sink = self._make("nveglglessink", f"sink_{safe}")

        detector_q = self._make("queue", f"detect_latest_{safe}")
        detector_convert = self._make("nvvideoconvert", f"detect_export_{safe}")
        detector_caps = self._make("capsfilter", f"detect_caps_{safe}")
        detector_sink = self._make("appsink", f"detect_sink_{safe}")

        for queue in (source_q, display_q, detector_q):
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
        self._set_if(source, "select-rtp-protocol", 4)  # Frozen Step1 TCP baseline.
        self._set_if(source, "rtsp-reconnect-interval", self.reconnect_sec)
        self._set_if(source, "rtsp-reconnect-attempts", -1)
        self._set_if(source, "message-forward", True)
        self._set_if(source, "async-handling", True)

        # Frozen Step1 display resize happens exactly once and remains NVMM/device.
        self._set_if(shared_convert, "gpu-id", self.gpu_id)
        self._set_if(shared_convert, "nvbuf-memory-type", 0)
        self._set_if(shared_convert, "interpolation-method", self.interpolation)
        shared_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM),format=NV12,width={self.tile_width},height={self.tile_height}"
            ),
        )

        self._set_if(display_sink, "sync", False)
        self._set_if(display_sink, "qos", False)
        self._set_if(display_sink, "async", False)
        self._set_if(display_sink, "enable-last-sample", False)
        self._set_if(display_sink, "force-aspect-ratio", False)

        # Detector now starts from the already-scaled 640x360 NVMM frame. Only a
        # small 1.05x resize/color export occurs on the specifically requested frame.
        self._set_if(detector_convert, "gpu-id", self.gpu_id)
        self._set_if(detector_convert, "nvbuf-memory-type", 0)
        self._set_if(detector_convert, "compute-hw", 1)
        self._set_if(detector_convert, "interpolation-method", 2)
        detector_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,width={DETECT_W},height={DETECT_CONTENT_H},"
                "pixel-aspect-ratio=1/1"
            ),
        )
        self._set_if(detector_sink, "emit-signals", True)
        self._set_if(detector_sink, "sync", False)
        self._set_if(detector_sink, "async", False)
        self._set_if(detector_sink, "drop", True)
        self._set_if(detector_sink, "max-buffers", 1)
        self._set_if(detector_sink, "enable-last-sample", False)
        self._set_if(detector_sink, "wait-on-eos", False)

        elements = (
            source,
            source_q,
            shared_convert,
            shared_caps,
            tee,
            display_q,
            display_sink,
            detector_q,
            detector_convert,
            detector_caps,
            detector_sink,
        )
        for element in elements:
            pipeline.add(element)

        self._require_link(source_q, shared_convert, f"{cid}:source_q->shared_convert")
        self._require_link(shared_convert, shared_caps, f"{cid}:shared_convert->caps")
        self._require_link(shared_caps, tee, f"{cid}:shared_caps->tee")
        self._link_tee_to_queue(tee, display_q, f"{cid}:tee->display")
        self._link_tee_to_queue(tee, detector_q, f"{cid}:tee->detector")
        self._require_link(display_q, display_sink, f"{cid}:display_q->sink")
        self._require_link(detector_q, detector_convert, f"{cid}:detector_q->convert")
        self._require_link(detector_convert, detector_caps, f"{cid}:detector_convert->caps")
        self._require_link(detector_caps, detector_sink, f"{cid}:detector_caps->appsink")

        display_sink_pad = display_sink.get_static_pad("sink")
        if display_sink_pad is None:
            raise RuntimeError(f"{cid}: display sink pad missing")
        display_sink_pad.add_probe(self.Gst.PadProbeType.BUFFER, self._render_probe, cid)

        detector_q_src = detector_q.get_static_pad("src")
        if detector_q_src is None:
            raise RuntimeError(f"{cid}: detector queue src pad missing")
        detector_q_src.add_probe(self.Gst.PadProbeType.BUFFER, self._detector_gate_probe, cid)
        detector_sink.connect("new-sample", self._on_detector_sample, cid)

        try:
            self.GstVideo.VideoOverlay.set_window_handle(
                display_sink, int(self.wall.children[index])
            )
            overlay_ok = 1
        except Exception as exc:
            raise RuntimeError(f"{cid}: GstVideoOverlay window binding failed: {exc}") from exc

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, cid)

        # Keep all inherited Step1/Step2 statistics pointed at the equivalent nodes.
        self.pipelines[cid] = pipeline
        self.sources[cid] = source
        self.queues[cid] = source_q
        self.converters[cid] = shared_convert
        self.capsfilters[cid] = shared_caps
        self.sinks[cid] = display_sink
        self.tees[cid] = tee
        self.display_output_queues[cid] = display_q
        self.detector_queues[cid] = detector_q
        self.detector_converts[cid] = detector_convert
        self.detector_sinks[cid] = detector_sink
        self.capture_requested[cid] = False
        self.det_capture_counts[cid] = 0
        self.det_result_counts[cid] = 0
        self.det_last_boxes[cid] = 0
        self.det_conversion_age_ms[cid] = deque(maxlen=2048)
        self.det_result_age_ms[cid] = deque(maxlen=2048)
        self.det_queue_qmax[cid] = 0

        print(
            "CAMERA_V11_STEP1V7_WINDOW "
            f"camera={cid} transport=tcp low_latency={requested_lowlat} "
            f"xid={self.wall.children[index]} overlay={overlay_ok} "
            f"tile={self.tile_width}x{self.tile_height}",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP2V7_WINDOW "
            f"camera={cid} shared_scale={self.tile_width}x{self.tile_height}/NVMM "
            f"detector_source=shared display_q=latest1 detector_q=latest1 "
            f"detector_output={DETECT_W}x{DETECT_CONTENT_H}/BGRx-RAW",
            flush=True,
        )


def main() -> int:
    return V11Step2DetectorB1SharedScaleV7().run()


if __name__ == "__main__":
    raise SystemExit(main())
