from __future__ import annotations

import signal

from services.ml_service.app.detector_substream import substream_uri
from services.ml_service.app.detector_substream_rtp_pts import DetectorSubstreamRtpPtsService
from services.ml_service.app.trt86_detector import CONTENT_H, INPUT_W


class DetectorSubstreamPrequeuePtsService(DetectorSubstreamRtpPtsService):
    """V11 RTP-timestamp PTS gate moved ahead of the leaky input queue.

    In V11 the source-stat probe was attached to input_q:sink, but the sparse PTS
    gate was attached to input_q:src. input_q is deliberately max-size-buffers=1
    and leaky=downstream, so a bursty RTSP source can discard decoded buffers before
    the PTS gate ever observes them. CAM-02 then reports ~10 FPS at the queue sink
    while the gate sees an irregular subset and falls below the requested 2 Hz.

    V12 installs source stats first and the PTS gate second on input_q:sink. GStreamer
    runs data probes on a pad in registration order; source stats therefore observe
    every incoming decoded frame, then the gate either drops it immediately or lets
    the selected sparse frame enter the queue. The leaky queue is retained as a
    safety boundary, but now only ~2 selected frames/s reach it and nvvideoconvert.
    """

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

        # Registration order matters: stats first sees every source buffer; gate
        # second decides whether that same buffer may enter the leaky queue.
        input_sink = input_q.get_static_pad("sink")
        input_sink.add_probe(self.Gst.PadProbeType.BUFFER, self._source_probe, cid)
        input_sink.add_probe(self.Gst.PadProbeType.BUFFER, self._capture_gate_probe, cid)

        source.connect("pad-added", self._source_pad_added, input_q, cid)
        self.sources[cid] = source
        self.input_queues[cid] = input_q
        print(
            f"ML_SUBSTREAM_GATE_POSITION {cid} pad=input_q:sink before_leaky_queue=1 "
            "source_stats_before_gate=1",
            flush=True,
        )


def main() -> int:
    service = DetectorSubstreamPrequeuePtsService()

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
