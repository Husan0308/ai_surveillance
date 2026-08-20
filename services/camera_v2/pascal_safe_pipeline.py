from __future__ import annotations

"""Pascal-safe RF-DETR camera runtime with no gst-nvtracker dependency.

The deployment GPU is a GTX 1050 Ti (Pascal). DeepStream 7.1 does not list
Pascal in its validated dGPU matrix and the hardware smoke log shows NvDCF
accepting the first mux batch and then stalling downstream. This runtime keeps
only the stages that are proven healthy on that machine:

RTSP -> NVDEC -> nvstreammux -> RF-DETR side capture -> motion predictor
     -> nvmultistreamtiler -> NVMM RGBA -> nvdsosd -> display sink

The primary display sink is nveglglessink. If an X11 session is driven by an
onboard/iGPU and EGL receives buffers but renders zero frames, the controller can
restart this same runtime in the bounded ximagesink fallback. Only the already-
composited 2x3 wall is downloaded to system memory in that fallback, never the six
individual camera streams.

It intentionally does not import tracker classes and never resolves or loads the
DeepStream low-level tracker library or NvDCF configuration files.
"""

import os
import time

import numpy as np

from .rfdetr_backend import install as _install_rfdetr_backend

_install_rfdetr_backend()

from .detection import CameraDetectionV2


class CameraPascalSafeRuntime(CameraDetectionV2):
    """Dedicated GTX 10-series runtime: RF-DETR + bounded motion prediction."""

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
        super().__init__()
        self.source_track_counts = {
            int(source_id): 0 for source_id in self.camera_index.values()
        }
        self.tracker_backend = "motion-predictor"
        self.tracker = None

    def _preflight(self) -> None:
        super()._preflight()
        if self.display_backend == "x11":
            if self.Gst.ElementFactory.find("ximagesink") is None:
                raise RuntimeError("ximagesink is unavailable for X11 display fallback")
            if self.Gst.ElementFactory.find("nvvideoconvert") is None:
                raise RuntimeError("nvvideoconvert is unavailable for X11 display fallback")

    def _make(self, factory: str, name: str):
        # DynamicCameraWallV2 creates its final display sink through this method.
        # The fallback changes only that final sink. All upstream DeepStream
        # decode/mux/tiler/OSD stages remain identical.
        if factory == "nveglglessink" and self.display_backend == "x11":
            factory = "ximagesink"
        return super()._make(factory, name)

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
        """Insert OSD without relying on gst-nvtracker.

        Gst.Element.unlink() has no boolean return value. Detach the baseline sink,
        verify both pads are free, then build the selected presentation chain.
        """

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
            # Hybrid Intel/NVIDIA X11 fallback: download only the already tiled
            # wall to ordinary BGRx system memory before ximagesink. This is more
            # expensive than EGL but avoids a permanent black window when the X
            # server is not owned by the NVIDIA GPU.
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
        mux_src.add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._pascal_mux_probe,
        )
        osd_src.add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._pascal_wall_probe,
        )
        final_sink_pad.add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._pascal_sink_probe,
        )
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
                active = 0
                for track in boxes.tracks.get(cid, {}).values():
                    if now - float(track.last_det_t) <= float(boxes.max_age):
                        active += 1
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

        # Do not blame EGL until upstream flow is proven: camera buffers, mux
        # batches, OSD output and buffers at the sink pad must all be advancing.
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
                f"from=egl to=x11 reason=zero-render "
                f"source_frames={source_frames} mux_batches={self.safe_mux_batches} "
                f"wall_frames={self.safe_wall_frames} sink_buffers={self.safe_sink_buffers}",
                flush=True,
            )
            self.stop()
            return False

        # If upstream itself is stalled, keep the process alive and let the stage
        # counters identify the actual failing component instead of masking it by
        # switching sinks.
        return True

    def _print_stats(self) -> bool:
        keep = CameraDetectionV2._print_stats(self)
        counts = self.live_source_counts()
        rendered, dropped = self._sink_stats()
        print(
            "CAMERA_PASCAL_SAFE "
            f"display={self.display_backend} "
            f"mux_batches={self.safe_mux_batches} "
            f"wall_frames={self.safe_wall_frames} "
            f"sink_buffers={self.safe_sink_buffers} "
            f"tracked_now={self.tracked_now} source_counts={counts} "
            f"rendered={rendered if rendered is not None else '?'} "
            f"dropped={dropped if dropped is not None else '?'} "
            "nvtracker=0 tracker=motion-predictor",
            flush=True,
        )
        return keep

    def run(self) -> int:
        self.display_watch_started = time.monotonic()
        if self.display_backend == "egl":
            self.GLib.timeout_add(1000, self._display_watchdog)
        print(
            "CAMERA_PASCAL_SAFE ready backend=RF-DETR-S "
            f"display={self.display_backend} "
            "tracker=motion-predictor nvtracker=disabled "
            "path=RTSP-NVDEC-mux-tiler-OSD-display",
            flush=True,
        )
        return super().run()


def main() -> int:
    enabled = os.environ.get("CAMERA_V2_PASCAL_SAFE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        raise RuntimeError("CAMERA_V2_PASCAL_SAFE=1 is required")
    return CameraPascalSafeRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
