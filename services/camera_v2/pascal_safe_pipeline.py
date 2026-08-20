from __future__ import annotations

"""Pascal-safe RF-DETR camera runtime with display-first source isolation.

The deployment GPU is a GTX 1050 Ti (Pascal). The production graph therefore
keeps gst-nvtracker/NvDCF out of the hot path. More importantly, camera ingest is
never split before nvstreammux: the six RTSP/NVDEC sources use the same direct
source -> queue -> nvstreammux path that already proved stable on the target.

RF-DETR receives per-camera frames only after muxing:

RTSP/NVDEC -> nvstreammux -> tee -> display branch -> tiler -> OSD -> sink
                              \\-> nvstreamdemux -> per-camera RF-DETR capture

This makes detector startup/backpressure incapable of preventing a camera from
reaching nvstreammux. The primary display is nveglglessink; when upstream flow is
proven but EGL renders zero frames, the controller may restart once with the
bounded X11 system-memory fallback.
"""

import os
import time

import numpy as np

from .rfdetr_backend import install as _install_rfdetr_backend

_install_rfdetr_backend()

from .detection import CameraDetectionV2, INFER_HEIGHT, INFER_WIDTH
from .secure import SecureCameraWallV2


class CameraPascalSafeRuntime(CameraDetectionV2):
    """RF-DETR + motion prediction without any DeepStream tracker dependency."""

    def __init__(self) -> None:
        backend = os.environ.get("CAMERA_V2_DISPLAY_BACKEND", "egl").strip().lower()
        self.display_backend = backend if backend in {"egl", "x11"} else "egl"
        self.display_failover_requested = False
        self.display_watch_started = 0.0
        self.safe_wall_frames = 0
        self.safe_mux_batches = 0
        self.safe_sink_buffers = 0
        self.source_track_counts: dict[int, int] = {}
        self.tracked_now = 0
        self._safe_stride_logged: set[str] = set()
        self._startup_stall_reported = False
        super().__init__()
        self.source_track_counts = {
            int(source_id): 0 for source_id in self.camera_index.values()
        }
        self.tracker_backend = "motion-predictor"
        self.tracker = None

    def _preflight(self) -> None:
        super()._preflight()
        for plugin in ("tee", "nvstreamdemux", "nvvideoconvert", "appsink"):
            if self.Gst.ElementFactory.find(plugin) is None:
                raise RuntimeError(f"required Pascal-safe plugin is unavailable: {plugin}")
        if self.display_backend == "x11" and self.Gst.ElementFactory.find("ximagesink") is None:
            raise RuntimeError("ximagesink is unavailable for X11 display fallback")

    def _make(self, factory: str, name: str):
        if factory == "nveglglessink" and self.display_backend == "x11":
            factory = "ximagesink"
        return super()._make(factory, name)

    def _add_camera(self, index, camera) -> None:
        """Keep the proven direct source -> queue -> mux path.

        CameraDetectionV2 normally inserts a tee before nvstreammux. On this
        deployment that branch is deliberately removed; inference is attached
        after the mux in _install_postmux_inference().
        """

        cid = camera.camera_id
        self.camera_index[cid] = int(index)
        self.capture_requested[cid] = False
        SecureCameraWallV2._add_camera(self, index, camera)

    @staticmethod
    def _queue_latest(owner, element, buffers: int = 2) -> None:
        owner._set_if(element, "max-size-buffers", max(1, int(buffers)))
        owner._set_if(element, "max-size-bytes", 0)
        owner._set_if(element, "max-size-time", 0)
        owner._set_if(element, "leaky", 2)
        owner._set_if(element, "silent", True)

    def _request_src_pad(self, element, name: str):
        request_simple = getattr(element, "request_pad_simple", None)
        pad = request_simple(name) if request_simple else None
        if pad is None:
            pad = element.get_request_pad(name)
        if pad is None:
            raise RuntimeError(f"{element.get_name()} could not allocate {name}")
        self.tee_request_pads.append((element, pad))
        return pad

    def _install_postmux_inference(self) -> None:
        """Split the batched mux output, never individual source inputs."""

        mux_src = self.mux.get_static_pad("src")
        tiler_sink = self.tiler.get_static_pad("sink")
        if mux_src is None or tiler_sink is None:
            raise RuntimeError("could not inspect nvstreammux -> tiler link")
        if mux_src.is_linked():
            self.mux.unlink(self.tiler)
        if mux_src.is_linked() or tiler_sink.is_linked():
            raise RuntimeError("could not detach nvstreammux -> tiler")

        tee = self._make("tee", "pascal_postmux_tee")
        display_q = self._make("queue", "pascal_display_branch")
        infer_q = self._make("queue", "pascal_infer_batch_branch")
        demux = self._make("nvstreamdemux", "pascal_infer_demux")
        self._queue_latest(self, display_q, 2)
        self._queue_latest(self, infer_q, 1)

        for element in (tee, display_q, infer_q, demux):
            self.pipeline.add(element)

        if not self.mux.link(tee):
            raise RuntimeError("failed nvstreammux -> postmux tee")

        tee_display = self._request_src_pad(tee, "src_%u")
        tee_infer = self._request_src_pad(tee, "src_%u")
        if tee_display.link(display_q.get_static_pad("sink")) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError("failed postmux tee -> display queue")
        if tee_infer.link(infer_q.get_static_pad("sink")) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError("failed postmux tee -> inference queue")
        if not display_q.link(self.tiler):
            raise RuntimeError("failed display queue -> nvmultistreamtiler")
        if not infer_q.link(demux):
            raise RuntimeError("failed inference queue -> nvstreamdemux")

        self.postmux_tee = tee
        self.postmux_display_queue = display_q
        self.postmux_infer_queue = infer_q
        self.infer_demux = demux

        for index, camera in enumerate(self.cameras):
            cid = camera.camera_id
            queue = self._make("queue", f"detect_queue_{index}")
            converter = self._make("nvvideoconvert", f"detect_convert_{index}")
            capsfilter = self._make("capsfilter", f"detect_caps_{index}")
            appsink = self._make("appsink", f"detect_sink_{index}")

            self._queue_latest(self, queue, 1)
            self._set_if(converter, "gpu-id", self.gpu_id)
            self._set_if(converter, "compute-hw", 1)
            self._set_if(converter, "interpolation-method", 2)
            capsfilter.set_property(
                "caps",
                self.Gst.Caps.from_string(
                    "video/x-raw,format=BGRx,"
                    f"width={INFER_WIDTH},height={INFER_HEIGHT},pixel-aspect-ratio=1/1"
                ),
            )
            appsink.set_property("emit-signals", True)
            appsink.set_property("sync", False)
            appsink.set_property("drop", True)
            appsink.set_property("max-buffers", 1)
            self._set_if(appsink, "enable-last-sample", False)
            self._set_if(appsink, "wait-on-eos", False)

            for element in (queue, converter, capsfilter, appsink):
                self.pipeline.add(element)

            demux_pad = self._request_src_pad(demux, f"src_{index}")
            if demux_pad.link(queue.get_static_pad("sink")) != self.Gst.PadLinkReturn.OK:
                raise RuntimeError(f"{cid}: nvstreamdemux -> inference queue failed")
            if not queue.link(converter):
                raise RuntimeError(f"{cid}: inference queue -> convert failed")
            if not converter.link(capsfilter):
                raise RuntimeError(f"{cid}: inference convert -> caps failed")
            if not capsfilter.link(appsink):
                raise RuntimeError(f"{cid}: inference caps -> appsink failed")

            queue.get_static_pad("src").add_probe(
                self.Gst.PadProbeType.BUFFER,
                self._infer_gate_probe,
                cid,
            )
            appsink.connect("new-sample", self._on_infer_sample, cid)

        print(
            "CAMERA_DETECT_PATH mode=postmux-demux "
            "source_path=direct-to-nvstreammux detector_cannot-block-ingest=1",
            flush=True,
        )

    def _on_infer_sample(self, sink, cid: str):
        """Copy BGRx using the actual mapped row stride, not width*4."""

        sample = sink.emit("pull-sample")
        if sample is None:
            with self.capture_lock:
                self.capture_requested[cid] = True
            return self.Gst.FlowReturn.OK

        structure = sample.get_caps().get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            with self.capture_lock:
                self.capture_requested[cid] = True
            return self.Gst.FlowReturn.OK

        try:
            tight_stride = width * 4
            mapped_size = int(getattr(mapped, "size", len(mapped.data)))
            if mapped_size < tight_stride * height:
                raise RuntimeError(
                    f"{cid}: BGRx buffer too small: {mapped_size} < {tight_stride * height}"
                )
            row_stride = (
                mapped_size // height
                if height > 0 and mapped_size % height == 0
                else tight_stride
            )
            if row_stride < tight_stride:
                raise RuntimeError(
                    f"{cid}: invalid BGRx stride={row_stride}, tight={tight_stride}"
                )

            needed = row_stride * height
            raw = np.frombuffer(mapped.data, dtype=np.uint8, count=needed)
            rows = raw.reshape((height, row_stride))
            bgrx = rows[:, :tight_stride].reshape((height, width, 4))
            frame = bgrx[..., :3].copy()

            if cid not in self._safe_stride_logged:
                self._safe_stride_logged.add(cid)
                print(
                    f"CAMERA_INFER_LAYOUT {cid} size={mapped_size} "
                    f"frame={width}x{height} stride={row_stride} tight={tight_stride}",
                    flush=True,
                )
        finally:
            buffer.unmap(mapped)

        self.mailbox.put(cid, time.monotonic(), frame)
        return self.Gst.FlowReturn.OK

    def _install_osd_and_meta(self) -> None:
        self._install_postmux_inference()

        queue_src = self.wall_queue.get_static_pad("src")
        sink_pad = self.sink.get_static_pad("sink")
        if queue_src is None or sink_pad is None:
            raise RuntimeError("could not inspect baseline wall -> display pads")
        if queue_src.is_linked():
            self.wall_queue.unlink(self.sink)
        if queue_src.is_linked() or sink_pad.is_linked():
            raise RuntimeError("could not detach baseline wall -> display link")

        convert = self._make("nvvideoconvert", "pascal_wall_convert")
        caps = self._make("capsfilter", "pascal_wall_caps")
        osd = self._make("nvdsosd", "pascal_osd")
        self._set_if(convert, "gpu-id", self.gpu_id)
        self._set_if(convert, "compute-hw", 1)
        caps.set_property(
            "caps",
            self.Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
        )
        self._set_if(osd, "process-mode", 1)
        self._set_if(osd, "display-bbox", True)
        self._set_if(osd, "display-text", False)
        self._set_if(osd, "display-mask", False)
        self._set_if(osd, "gpu-id", self.gpu_id)
        for element in (convert, caps, osd):
            self.pipeline.add(element)

        if not self.wall_queue.link(convert):
            raise RuntimeError("failed wall queue -> nvvideoconvert")
        if not convert.link(caps):
            raise RuntimeError("failed nvvideoconvert -> RGBA caps")
        if not caps.link(osd):
            raise RuntimeError("failed RGBA caps -> nvdsosd")

        if self.display_backend == "egl":
            if not osd.link(self.sink):
                raise RuntimeError("failed nvdsosd -> nveglglessink")
        else:
            download = self._make("nvvideoconvert", "pascal_x11_download")
            sys_caps = self._make("capsfilter", "pascal_x11_caps")
            self._set_if(download, "gpu-id", self.gpu_id)
            self._set_if(download, "compute-hw", 1)
            sys_caps.set_property(
                "caps",
                self.Gst.Caps.from_string("video/x-raw,format=BGRx"),
            )
            self.pipeline.add(download)
            self.pipeline.add(sys_caps)
            if not osd.link(download):
                raise RuntimeError("failed nvdsosd -> X11 download convert")
            if not download.link(sys_caps):
                raise RuntimeError("failed X11 download convert -> system caps")
            if not sys_caps.link(self.sink):
                raise RuntimeError("failed system BGRx -> ximagesink")
            self.x11_download = download
            self.x11_caps = sys_caps

        mux_src = self.mux.get_static_pad("src")
        osd_src = osd.get_static_pad("src")
        final_sink_pad = self.sink.get_static_pad("sink")
        if mux_src is None or osd_src is None or final_sink_pad is None:
            raise RuntimeError("could not obtain mux/OSD/display probe pads")
        mux_src.add_probe(self.Gst.PadProbeType.BUFFER, self._pascal_mux_probe)
        osd_src.add_probe(self.Gst.PadProbeType.BUFFER, self._pascal_wall_probe)
        final_sink_pad.add_probe(self.Gst.PadProbeType.BUFFER, self._pascal_sink_probe)
        self.osd = osd

    def _pascal_mux_probe(self, pad, info):
        self.safe_mux_batches += 1
        return CameraDetectionV2._inject_boxes_probe(self, pad, info)

    def _pascal_wall_probe(self, pad, info):
        self.safe_wall_frames += 1
        return CameraDetectionV2._wall_probe(self, pad, info)

    def _pascal_sink_probe(self, _pad, _info):
        self.safe_sink_buffers += 1
        return self.Gst.PadProbeReturn.OK

    def _active_motion_counts(self) -> dict[int, int]:
        now = time.monotonic()
        output = {int(source_id): 0 for source_id in self.camera_index.values()}
        boxes = getattr(self, "boxes", None)
        if boxes is None:
            return output
        with boxes.lock:
            for cid, source_id in self.camera_index.items():
                active = sum(
                    1
                    for track in boxes.tracks.get(cid, {}).values()
                    if now - float(track.last_det_t) <= float(boxes.max_age)
                )
                output[int(source_id)] = active
        return output

    def live_source_counts(self) -> dict[int, int]:
        counts = self._active_motion_counts()
        with self.det_lock:
            self.source_track_counts = counts
            self.tracked_now = sum(counts.values())
        return dict(counts)

    def _display_watchdog(self) -> bool:
        if self.display_backend != "egl" or self._stopping:
            return False
        elapsed = time.monotonic() - self.display_watch_started
        if elapsed < float(os.environ.get("CAMERA_V2_EGL_FAILOVER_SEC", "8.0")):
            return True

        source_frames = sum(int(stat.frames) for stat in self.stats.values())
        rendered, _dropped = self._sink_stats()
        if (
            source_frames >= 60
            and self.safe_mux_batches >= 10
            and self.safe_wall_frames >= 10
            and self.safe_sink_buffers >= 10
            and rendered == 0
        ):
            self.display_failover_requested = True
            print(
                "CAMERA_DISPLAY_FAILOVER "
                f"from=egl to=x11 reason=zero-render source_frames={source_frames} "
                f"mux_batches={self.safe_mux_batches} wall_frames={self.safe_wall_frames} "
                f"sink_buffers={self.safe_sink_buffers}",
                flush=True,
            )
            self.stop()
            return False
        return True

    def _startup_watchdog(self) -> bool:
        if self._stopping:
            return False
        if time.monotonic() - self.display_watch_started < 10.0:
            return True
        if self.safe_mux_batches > 0:
            return False
        source_counts = {
            cid: int(stat.frames) for cid, stat in self.stats.items()
        }
        stage = "source-or-auth" if sum(source_counts.values()) == 0 else "nvstreammux"
        if not self._startup_stall_reported:
            self._startup_stall_reported = True
            print(
                "CAMERA_STARTUP_STALL "
                f"stage={stage} source_frames={source_counts} mux_batches=0",
                flush=True,
            )
        return True

    def _print_stats(self) -> bool:
        keep = CameraDetectionV2._print_stats(self)
        counts = self.live_source_counts()
        rendered, dropped = self._sink_stats()
        source_total = sum(int(stat.frames) for stat in self.stats.values())
        print(
            "CAMERA_PASCAL_SAFE "
            f"display={self.display_backend} source_frames={source_total} "
            f"mux_batches={self.safe_mux_batches} wall_frames={self.safe_wall_frames} "
            f"sink_buffers={self.safe_sink_buffers} tracked_now={self.tracked_now} "
            f"source_counts={counts} rendered={rendered if rendered is not None else '?'} "
            f"dropped={dropped if dropped is not None else '?'} "
            "nvtracker=0 tracker=motion-predictor detector_path=postmux-demux",
            flush=True,
        )
        return keep

    def run(self) -> int:
        self.display_watch_started = time.monotonic()
        self.GLib.timeout_add(1000, self._startup_watchdog)
        if self.display_backend == "egl":
            self.GLib.timeout_add(1000, self._display_watchdog)
        print(
            "CAMERA_PASCAL_SAFE ready backend=RF-DETR-S "
            f"display={self.display_backend} tracker=motion-predictor nvtracker=disabled "
            "source_path=direct-to-mux detector_path=postmux-demux",
            flush=True,
        )
        return super().run()


def main() -> int:
    enabled = os.environ.get("CAMERA_V2_PASCAL_SAFE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled:
        raise RuntimeError("CAMERA_V2_PASCAL_SAFE=1 is required")
    return CameraPascalSafeRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
