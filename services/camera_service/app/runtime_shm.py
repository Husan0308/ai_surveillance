from __future__ import annotations

import sys
import time

from .runtime import CameraServiceRuntime
from .shm_frame import FrameDemandOwner, LatestFrameMmapWriter


ML_W = 672
ML_H = 378


class CameraServiceShmRuntime(CameraServiceRuntime):
    """AI-free camera service with demand-driven latest-frame IPC.

    Per camera:
        RTSP/NVDEC -> tee
          -> latest-only full-rate drain/display path
          -> latest-only ML queue -> demand gate -> GPU resize/color convert
             -> RAW BGR appsink -> double-buffered /dev/shm latest frame

    Expensive ML conversion only runs when the ML service requests a frame.
    With one sequential ML scheduler, conversion and TensorRT do not intentionally
    overlap on the GPU and no unused 2 Hz frames are produced.
    """

    def __init__(self) -> None:
        self.ml_publish_counts: dict[str, int] = {}
        self.ml_publish_last: dict[str, int] = {}
        self.ml_publish_stat_at = time.monotonic()
        self.ml_writers: dict[str, LatestFrameMmapWriter] = {}
        self.ml_demands: dict[str, FrameDemandOwner] = {}
        self.ml_served_request: dict[str, int] = {}
        self.ml_reserved_request: dict[str, int] = {}
        self.ml_capture_started_ns: dict[str, int] = {}
        self.ml_inflight: dict[str, bool] = {}
        self.tees: dict[str, object] = {}
        self.ml_queues: dict[str, object] = {}
        self.ml_sinks: dict[str, object] = {}
        self.tee_request_pads: list[tuple[object, object]] = []
        super().__init__()
        print(
            "CAMERA_SERVICE_SHM enabled=1 tap=672x378x3 mode=demand-jit "
            f"headless={int(self.headless)} policy=request-next-frame "
            "gate-before-convert consumer_backpressure=0",
            flush=True,
        )

    def _request_tee_pad(self, tee):
        request_simple = getattr(tee, "request_pad_simple", None)
        pad = request_simple("src_%u") if request_simple else None
        if pad is None:
            pad = tee.get_request_pad("src_%u")
        if pad is None:
            raise RuntimeError(f"could not allocate tee pad for {tee.get_name()}")
        self.tee_request_pads.append((tee, pad))
        return pad

    def _link_tee_to_queue(self, tee, queue, label: str) -> None:
        src = self._request_tee_pad(tee)
        sink = queue.get_static_pad("sink")
        if sink is None or src.link(sink) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"failed to link {label}")

    def _add_camera(self, index, camera) -> None:
        source = self._make("nvurisrcbin", f"camera_service_source_{index}")
        tee = self._make("tee", f"camera_service_tee_{index}")
        display_q = self._make("queue", f"camera_service_q_{index}")
        ml_q = self._make("queue", f"camera_service_ml_q_{index}")
        ml_convert = self._make("nvvideoconvert", f"camera_service_ml_convert_{index}")
        ml_caps = self._make("capsfilter", f"camera_service_ml_caps_{index}")
        ml_sink = self._make("appsink", f"camera_service_ml_sink_{index}")
        drain_sink = None

        self._latest_queue(display_q)
        self._latest_queue(ml_q)

        source.connect("deep-element-added", self._configure_rtsp_child, camera)
        source.set_property("uri", camera.uri)
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "select-rtp-protocol", 4 if self.settings.rtsp_transport == "tcp" else 0)
        self._set_if(source, "latency", self.settings.latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "num-extra-surfaces", self.settings.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "rtsp-reconnect-interval", 2)
        self._set_if(source, "rtsp-reconnect-attempts", 3)
        self._set_if(source, "message-forward", True)
        self._set_if(source, "async-handling", True)
        self._set_if(source, "gpu-id", self.settings.gpu_id)

        self._set_if(ml_convert, "gpu-id", self.settings.gpu_id)
        self._set_if(ml_convert, "compute-hw", 1)
        self._set_if(ml_convert, "interpolation-method", 2)
        ml_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=BGR,width={ML_W},height={ML_H},pixel-aspect-ratio=1/1"
            ),
        )
        self._set_if(ml_sink, "emit-signals", True)
        self._set_if(ml_sink, "sync", False)
        self._set_if(ml_sink, "max-buffers", 1)
        self._set_if(ml_sink, "drop", True)
        self._set_if(ml_sink, "enable-last-sample", False)
        self._set_if(ml_sink, "wait-on-eos", False)
        ml_sink.connect("new-sample", self._on_ml_sample, camera.camera_id)

        elements = [source, tee, display_q, ml_q, ml_convert, ml_caps, ml_sink]
        if self.headless:
            drain_sink = self._make("fakesink", f"camera_service_fakesink_{index}")
            self._configure_fakesink(drain_sink)
            elements.append(drain_sink)
        for element in elements:
            self.pipeline.add(element)

        if self.headless:
            self._link(display_q, drain_sink, f"{camera.camera_id}:drain_q->fakesink")
            self.headless_sinks[camera.camera_id] = drain_sink
        else:
            mux_pad = self._request_mux_pad(index)
            if display_q.get_static_pad("src").link(mux_pad) != self.Gst.PadLinkReturn.OK:
                raise RuntimeError(f"{camera.camera_id}: display queue->mux failed")

        self._link_tee_to_queue(tee, display_q, f"{camera.camera_id}:tee->display")
        self._link_tee_to_queue(tee, ml_q, f"{camera.camera_id}:tee->ml")
        self._link(ml_q, ml_convert, f"{camera.camera_id}:ml_q->convert")
        self._link(ml_convert, ml_caps, f"{camera.camera_id}:convert->caps")
        self._link(ml_caps, ml_sink, f"{camera.camera_id}:caps->appsink")

        ml_q.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._ml_demand_gate_probe,
            camera.camera_id,
        )
        display_q.get_static_pad("sink").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._source_probe,
            camera.camera_id,
        )
        source.connect("pad-added", self._source_pad_added_to_tee, tee, camera.camera_id)

        cid = camera.camera_id
        self.sources[cid] = source
        self.queues[cid] = display_q
        self.tees[cid] = tee
        self.ml_queues[cid] = ml_q
        self.ml_sinks[cid] = ml_sink
        self.ml_publish_counts[cid] = 0
        self.ml_publish_last[cid] = 0
        self.ml_writers[cid] = LatestFrameMmapWriter(cid, ML_W, ML_H, 3)
        self.ml_demands[cid] = FrameDemandOwner(cid)
        self.ml_served_request[cid] = 0
        self.ml_reserved_request[cid] = 0
        self.ml_capture_started_ns[cid] = 0
        self.ml_inflight[cid] = False

    def _source_pad_added_to_tee(self, _source, pad, tee, cid: str) -> None:
        caps = pad.get_current_caps()
        if caps is None or caps.get_size() == 0:
            try:
                caps = pad.query_caps(None)
            except Exception:
                caps = None
        if caps is not None and caps.get_size() > 0 and not caps.is_any():
            try:
                media = str(caps.get_structure(0).get_name())
            except Exception:
                media = ""
            if media and not media.startswith("video/"):
                return
        sink = tee.get_static_pad("sink")
        if sink is None or sink.is_linked():
            return
        result = pad.link(sink)
        if result != self.Gst.PadLinkReturn.OK:
            print(
                f"CAMERA_SERVICE {cid} source->tee failed result={result}",
                file=sys.stderr,
                flush=True,
            )

    def _ml_demand_gate_probe(self, _pad, _info, cid: str):
        if self.ml_inflight.get(cid, False):
            return self.Gst.PadProbeReturn.DROP
        requested = self.ml_demands[cid].requested_seq()
        served = self.ml_served_request.get(cid, 0)
        if requested <= served:
            return self.Gst.PadProbeReturn.DROP

        self.ml_inflight[cid] = True
        self.ml_reserved_request[cid] = requested
        self.ml_capture_started_ns[cid] = time.monotonic_ns()
        return self.Gst.PadProbeReturn.OK

    def _on_ml_sample(self, sink, cid: str):
        sample = sink.emit("pull-sample")
        if sample is None:
            self.ml_inflight[cid] = False
            return self.Gst.FlowReturn.OK
        try:
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))
            if width != ML_W or height != ML_H:
                raise RuntimeError(
                    f"{cid}: ML tap geometry={width}x{height}, expected={ML_W}x{ML_H}"
                )
            buffer = sample.get_buffer()
            ok, mapped = buffer.map(self.Gst.MapFlags.READ)
            if not ok:
                raise RuntimeError(f"{cid}: ML tap map failed")
            try:
                tight = width * 3
                mapped_size = int(getattr(mapped, "size", len(mapped.data)))
                row_stride = mapped_size // height if mapped_size % height == 0 else tight
                if row_stride < tight:
                    raise RuntimeError(f"{cid}: invalid ML tap stride={row_stride} tight={tight}")
                raw = memoryview(mapped.data)
                if row_stride == tight:
                    payload = raw[: tight * height]
                else:
                    compact = bytearray(tight * height)
                    for row in range(height):
                        src = row * row_stride
                        dst = row * tight
                        compact[dst : dst + tight] = raw[src : src + tight]
                    payload = compact
                captured_ns = self.ml_capture_started_ns.get(cid, time.monotonic_ns())
                self.ml_writers[cid].publish(payload, captured_ns)
                self.ml_publish_counts[cid] += 1
                self.ml_served_request[cid] = self.ml_reserved_request[cid]
            finally:
                buffer.unmap(mapped)
        except Exception as exc:
            print(
                f"CAMERA_SERVICE_SHM_WARNING {cid} {type(exc).__name__}:{exc}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            self.ml_inflight[cid] = False
        return self.Gst.FlowReturn.OK

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        now = time.monotonic()
        elapsed = max(0.001, now - self.ml_publish_stat_at)
        parts = []
        for camera in self.cameras:
            cid = camera.camera_id
            total = self.ml_publish_counts.get(cid, 0)
            previous = self.ml_publish_last.get(cid, 0)
            hz = (total - previous) / elapsed
            self.ml_publish_last[cid] = total
            requested = self.ml_demands[cid].requested_seq()
            served = self.ml_served_request.get(cid, 0)
            parts.append(
                f"{cid}:{hz:.2f}Hz frame_seq={self.ml_writers[cid].seq} "
                f"req={requested}/{served} inflight={int(self.ml_inflight.get(cid, False))}"
            )
        self.ml_publish_stat_at = now
        print("CAMERA_SERVICE_SHM_STATS " + " | ".join(parts), flush=True)
        return keep

    def stop(self) -> None:
        already = self.stopping
        super().stop()
        if not already:
            for writer in self.ml_writers.values():
                try:
                    writer.close()
                except Exception:
                    pass
            for demand in self.ml_demands.values():
                try:
                    demand.close()
                except Exception:
                    pass
            for owner, pad in self.tee_request_pads:
                try:
                    owner.release_request_pad(pad)
                except Exception:
                    pass
            self.tee_request_pads.clear()


def main() -> int:
    return CameraServiceShmRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
