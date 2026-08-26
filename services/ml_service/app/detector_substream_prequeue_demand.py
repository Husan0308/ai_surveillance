from __future__ import annotations

import signal

from services.ml_service.app.detector_substream import substream_uri
from services.ml_service.app.detector_substream_burst import DetectorSubstreamBurstService
from services.ml_service.app.trt86_detector import CONTENT_H, INPUT_W


class DetectorSubstreamPrequeueDemandService(DetectorSubstreamBurstService):
    """Wall-clock demand scheduler with the sparse gate ahead of the leaky queue.

    V12 proved that moving the gate to input_q:sink is live-safe once appsink keeps
    async=false. It also proved that PTS-paced acceptance still mirrors CAM-02's
    bursty RTP delivery in wall time: ~8 FPS arrival windows yield ~1.6 Hz gate and
    ~12 FPS windows yield ~2.4 Hz gate. For surveillance inference we want a stable
    wall-clock target, not a target measured in bursty media arrival time.

    This service therefore combines the strongest pieces already validated:
      * V9 wall-clock, phase-staggered, demand-latched capture scheduler;
      * V12 prequeue gate so an armed demand sees the very next decoded buffer;
      * bounded application pending deque (default depth 4);
      * tcp-timestamp=false for RTP-interpolated timestamps/telemetry;
      * live/preroll-safe appsink properties.

    Only the demanded ~2 frames/s/camera pass into nvvideoconvert. The full decoded
    source rate is observed only by the lightweight pad probes before the queue.
    """

    def _configure_rtsp_child(self, _bin, _sub_bin, element, camera) -> None:
        super()._configure_rtsp_child(_bin, _sub_bin, element, camera)
        factory = element.get_factory()
        if factory is None or factory.get_name() != "rtspsrc":
            return
        if self.rtsp_transport == "tcp":
            self._set_if(element, "tcp-timestamp", False)
            print(
                f"ML_SUBSTREAM_RTP_CLOCK {camera.camera_id} transport=tcp "
                "tcp_timestamp=0 clock=rtp-interpolated",
                flush=True,
            )

    def _add_camera(self, index, camera) -> None:
        cid = camera.camera_id
        source = self._make("nvurisrcbin", f"ml_sub_source_{index}")
        input_q = self._make("queue", f"ml_sub_input_q_{index}")
        convert = self._make("nvvideoconvert", f"ml_sub_convert_{index}")
        caps = self._make("capsfilter", f"ml_sub_caps_{index}")
        output_q = self._make("queue", f"ml_sub_output_q_{index}")
        sink = self._make("appsink", f"ml_sub_sink_{index}")

        self._latest_queue(input_q)
        self._latest_queue(output_q)
        self._set_if(convert, "gpu-id", self.gpu_id)
        self._set_if(convert, "compute-hw", 1)
        self._set_if(convert, "interpolation-method", 2)
        caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,width={INPUT_W},height={CONTENT_H},pixel-aspect-ratio=1/1"
            ),
        )

        self._set_if(sink, "emit-signals", True)
        self._set_if(sink, "sync", False)
        self._set_if(sink, "max-buffers", 1)
        self._set_if(sink, "drop", True)
        self._set_if(sink, "wait-on-eos", False)
        self._set_if(sink, "enable-last-sample", False)
        self._set_if(sink, "async", False)
        self._set_if(sink, "qos", False)
        self._set_if(sink, "processing-deadline", 0)
        self._set_if(sink, "max-lateness", -1)
        sink.connect("new-sample", self._on_sample, cid)

        source.connect("deep-element-added", self._configure_rtsp_child, camera)
        source.set_property("uri", substream_uri(camera.uri))
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "select-rtp-protocol", 4 if self.rtsp_transport == "tcp" else 0)
        self._set_if(source, "latency", self.rtsp_latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "num-extra-surfaces", self.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "rtsp-reconnect-interval", 2)
        self._set_if(source, "rtsp-reconnect-attempts", 3)
        self._set_if(source, "message-forward", True)
        self._set_if(source, "async-handling", True)
        self._set_if(source, "gpu-id", self.gpu_id)

        for element in (source, input_q, convert, caps, output_q, sink):
            self.pipeline.add(element)
        self._link(input_q, convert, f"{cid}:input_q->convert")
        self._link(convert, caps, f"{cid}:convert->caps")
        self._link(caps, output_q, f"{cid}:caps->output_q")
        self._link(output_q, sink, f"{cid}:output_q->appsink")

        # Source telemetry first, then the demand gate. The gate is intentionally
        # on the queue sink so a max-size=1 leaky queue cannot discard the buffer
        # that should satisfy an armed wall-clock demand.
        input_sink = input_q.get_static_pad("sink")
        input_sink.add_probe(self.Gst.PadProbeType.BUFFER, self._source_probe, cid)
        input_sink.add_probe(self.Gst.PadProbeType.BUFFER, self._capture_gate_probe, cid)

        source.connect("pad-added", self._source_pad_added, input_q, cid)
        self.sources[cid] = source
        self.input_queues[cid] = input_q
        print(
            f"ML_SUBSTREAM_GATE_POSITION {cid} pad=input_q:sink before_leaky_queue=1 "
            "source_stats_before_gate=1 sink_async=0 scheduler=wall-demand-latched",
            flush=True,
        )


def main() -> int:
    service = DetectorSubstreamPrequeueDemandService()

    def stop(_signum, _frame) -> None:
        service.stop_requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        return service.run()
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
