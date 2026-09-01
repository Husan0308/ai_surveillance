from __future__ import annotations

import os
import sys

from services.camera_v2.native_bridge import NativeMetaBridge
from services.ml_service.app.config import CameraConfig

from .bbox_overlay_ipc_v1 import BboxStateReader
from .step1_cam02_lowlat_v7 import V11Step1Cam02LowLatV7


class V11Step1BboxOverlayV1(V11Step1Cam02LowLatV7):
    """V7 independent camera wall + latest-only local tracker bbox overlay.

    The six RTSP sources, TCP/latency/decoder policy, independent X11 windows,
    one-buffer leaky queues and nveglglessink remain the V7 design. Each independent
    display pipeline adds only a batch-size-1 nvstreammux to obtain standard
    NvDsFrameMeta, GPU nvvideoconvert + nvdsosd, and a read-only metadata probe.

    Detector/tracker remain in the existing separate Step3 process. No detector,
    tracker, ReID, face model or Python video copy is added to the display process.
    """

    def __init__(self) -> None:
        self.bbox_reader = BboxStateReader()
        self.bbox_bridge = NativeMetaBridge()
        self.bbox_stale_sec = max(
            0.60, min(2.0, float(os.environ.get("V11_BBOX_STALE_SEC", "1.10")))
        )
        self.bbox_muxers: dict[str, object] = {}
        self.bbox_osds: dict[str, object] = {}
        self.bbox_request_pads: dict[str, tuple[object, object]] = {}
        self.bbox_drawn: dict[str, int] = {}
        self.bbox_probe_errors = 0
        super().__init__()
        self.bbox_drawn = {camera.camera_id: 0 for camera in self.cameras}
        print(
            "CAMERA_V11_BBOX_DISPLAY_ARCH base=step1-v7 independent_pipelines=6 "
            "rtsp_changed=0 decoder_changed=0 queue_changed=0 bbox_ipc=latest-only "
            "mux=batch1-per-camera osd=gpu detector=0 tracker=0 reid=0 face=0",
            flush=True,
        )
        print(
            "CAMERA_V11_BBOX_DISPLAY_POLICY "
            f"stale_sec={self.bbox_stale_sec:.2f} prediction_max=0.45s "
            f"state={self.bbox_reader.path} labels=0",
            flush=True,
        )

    def _preflight(self) -> None:
        super()._preflight()
        required = ("nvstreammux", "nvdsosd")
        missing = [name for name in required if self.Gst.ElementFactory.find(name) is None]
        if missing:
            raise RuntimeError("V11 bbox display missing DeepStream plugins: " + ", ".join(missing))

    def _build_camera(self, index: int, camera: CameraConfig) -> None:
        cid = camera.camera_id
        requested_lowlat = int(cid in self.lowlat_cameras)
        safe = cid.lower().replace("-", "_")

        pipeline = self.Gst.Pipeline.new(f"v11_bbox_{safe}")
        if pipeline is None:
            raise RuntimeError(f"{cid}: could not create bbox pipeline")

        source = self._make("nvurisrcbin", f"source_{safe}")
        queue = self._make("queue", f"latest_{safe}")
        mux = self._make("nvstreammux", f"bbox_mux_{safe}")
        convert = self._make("nvvideoconvert", f"bbox_rgba_{safe}")
        capsfilter = self._make("capsfilter", f"bbox_caps_{safe}")
        osd = self._make("nvdsosd", f"bbox_osd_{safe}")
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
        self._set_if(source, "select-rtp-protocol", 4)  # preserve V7 TCP-only policy
        self._set_if(source, "rtsp-reconnect-interval", self.reconnect_sec)
        self._set_if(source, "rtsp-reconnect-attempts", -1)
        self._set_if(source, "message-forward", True)
        self._set_if(source, "async-handling", True)

        self._set_if(mux, "batch-size", 1)
        self._set_if(mux, "live-source", True)
        self._set_if(mux, "width", self.tile_width)
        self._set_if(mux, "height", self.tile_height)
        self._set_if(mux, "enable-padding", False)
        self._set_if(mux, "batched-push-timeout", 40_000)
        self._set_if(mux, "sync-inputs", False)
        self._set_if(mux, "buffer-pool-size", 4)
        self._set_if(mux, "nvbuf-memory-type", 0)
        self._set_if(mux, "gpu-id", self.gpu_id)
        self._set_if(mux, "interpolation-method", self.interpolation)

        self._set_if(convert, "gpu-id", self.gpu_id)
        self._set_if(convert, "nvbuf-memory-type", 0)
        capsfilter.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM),format=RGBA,width={self.tile_width},height={self.tile_height}"
            ),
        )
        self._set_if(osd, "process-mode", 1)  # GPU mode on dGPU
        self._set_if(osd, "display-bbox", True)
        self._set_if(osd, "display-text", False)

        self._set_if(sink, "sync", False)
        self._set_if(sink, "qos", False)
        self._set_if(sink, "async", False)
        self._set_if(sink, "enable-last-sample", False)
        self._set_if(sink, "force-aspect-ratio", False)

        for element in (source, queue, mux, convert, capsfilter, osd, sink):
            pipeline.add(element)

        queue_src = queue.get_static_pad("src")
        mux_sink = mux.request_pad_simple("sink_0") if hasattr(mux, "request_pad_simple") else None
        if mux_sink is None:
            mux_sink = mux.get_request_pad("sink_0")
        if queue_src is None or mux_sink is None:
            raise RuntimeError(f"{cid}: mux request pad missing")
        if queue_src.link(mux_sink) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: queue->mux pad link failed")

        self._require_link(mux, convert, f"{cid}:mux->rgba")
        self._require_link(convert, capsfilter, f"{cid}:rgba->caps")
        self._require_link(capsfilter, osd, f"{cid}:caps->osd")
        self._require_link(osd, sink, f"{cid}:osd->egl")

        mux_src = mux.get_static_pad("src")
        sink_pad = sink.get_static_pad("sink")
        if mux_src is None or sink_pad is None:
            raise RuntimeError(f"{cid}: bbox probe pad missing")
        mux_src.add_probe(self.Gst.PadProbeType.BUFFER, self._bbox_overlay_probe, cid)
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
        self.bbox_muxers[cid] = mux
        self.bbox_osds[cid] = osd
        self.bbox_request_pads[cid] = (mux, mux_sink)

        print(
            "CAMERA_V11_BBOX_WINDOW "
            f"camera={cid} transport=tcp low_latency={requested_lowlat} "
            f"xid={self.wall.children[index]} overlay={overlay_ok} "
            f"tile={self.tile_width}x{self.tile_height} mux_batch=1 osd_gpu=1",
            flush=True,
        )

    def _bbox_overlay_probe(self, _pad, info, cid: str):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        try:
            tracks = self.bbox_reader.camera_tracks(
                cid,
                stale_sec=self.bbox_stale_sec,
                width=self.tile_width,
                height=self.tile_height,
            )
            if tracks:
                # Each independent display pipeline has one mux source at pad/source 0.
                added = self.bbox_bridge.add_tracked_boxes(buffer, 0, tracks)
                if added > 0:
                    self.bbox_drawn[cid] = self.bbox_drawn.get(cid, 0) + int(added)
        except Exception as exc:
            # Drawing must be fail-open: camera rendering continues with no box.
            self.bbox_probe_errors += 1
            if self.bbox_probe_errors <= 3 or self.bbox_probe_errors % 200 == 0:
                print(
                    "CAMERA_V11_BBOX_OVERLAY warning="
                    f"{type(exc).__name__}:{exc} camera={cid} errors={self.bbox_probe_errors}",
                    file=sys.stderr,
                    flush=True,
                )
        return self.Gst.PadProbeReturn.OK

    def _print_stats(self):
        result = super()._print_stats()
        total = sum(self.bbox_drawn.values())
        per_camera = ",".join(f"{cid}:{self.bbox_drawn.get(cid, 0)}" for cid in sorted(self.bbox_drawn))
        print(
            "CAMERA_V11_BBOX_OVERLAY "
            f"drawn={total} per_camera={per_camera} ipc_errors={self.bbox_reader.errors} "
            f"probe_errors={self.bbox_probe_errors} stale_sec={self.bbox_stale_sec:.2f}",
            flush=True,
        )
        return result


def main() -> int:
    return V11Step1BboxOverlayV1().run()


if __name__ == "__main__":
    raise SystemExit(main())
