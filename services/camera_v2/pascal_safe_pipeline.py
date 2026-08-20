from __future__ import annotations

"""Pascal-safe RF-DETR camera runtime with no gst-nvtracker dependency.

The deployment GPU is a GTX 1050 Ti (Pascal). DeepStream 7.1 does not list
Pascal in its validated dGPU matrix and the hardware smoke log shows NvDCF
accepting the first mux batch and then stalling downstream. This runtime keeps
only the stages that are proven healthy on that machine:

RTSP -> NVDEC -> nvstreammux -> RF-DETR side capture -> motion predictor
     -> nvmultistreamtiler -> NVMM RGBA -> nvdsosd -> nveglglessink

It intentionally does not import the tracker classes and never resolves or loads
DeepStream's low-level tracker library or NvDCF configuration files.
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
        self.safe_wall_frames = 0
        self.safe_mux_batches = 0
        self.source_track_counts: dict[int, int] = {}
        self.tracked_now = 0
        self._safe_stride_logged: set[str] = set()
        super().__init__()
        self.source_track_counts = {
            int(source_id): 0 for source_id in self.camera_index.values()
        }
        self.tracker_backend = "motion-predictor"
        self.tracker = None

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

        Gst.Element.unlink() has no boolean return value. The previous fallback
        called ``if not wall_queue.unlink(sink)`` and therefore always raised.
        Detach first, verify pad state, then build the presentation chain.
        """

        queue_src = self.wall_queue.get_static_pad("src")
        sink_pad = self.sink.get_static_pad("sink")
        if queue_src is None or sink_pad is None:
            raise RuntimeError("could not inspect baseline wall -> EGL pads")

        if queue_src.is_linked():
            self.wall_queue.unlink(self.sink)
        if queue_src.is_linked() or sink_pad.is_linked():
            raise RuntimeError("could not detach baseline wall -> EGL link")

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
        if not osd.link(self.sink):
            raise RuntimeError("failed nvdsosd -> nveglglessink")

        mux_src = self.mux.get_static_pad("src")
        osd_src = osd.get_static_pad("src")
        if mux_src is None or osd_src is None:
            raise RuntimeError("could not obtain mux/OSD probe pads")
        mux_src.add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._pascal_mux_probe,
        )
        osd_src.add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._pascal_wall_probe,
        )
        self.osd = osd

    def _pascal_mux_probe(self, pad, info):
        self.safe_mux_batches += 1
        return CameraDetectionV2._inject_boxes_probe(self, pad, info)

    def _pascal_wall_probe(self, pad, info):
        self.safe_wall_frames += 1
        return CameraDetectionV2._wall_probe(self, pad, info)

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

    def _print_stats(self) -> bool:
        keep = CameraDetectionV2._print_stats(self)
        counts = self.live_source_counts()
        rendered, dropped = self._sink_stats()
        print(
            "CAMERA_PASCAL_SAFE "
            f"mux_batches={self.safe_mux_batches} "
            f"wall_frames={self.safe_wall_frames} "
            f"tracked_now={self.tracked_now} source_counts={counts} "
            f"rendered={rendered if rendered is not None else '?'} "
            f"dropped={dropped if dropped is not None else '?'} "
            "nvtracker=0 tracker=motion-predictor",
            flush=True,
        )
        return keep

    def run(self) -> int:
        print(
            "CAMERA_PASCAL_SAFE ready backend=RF-DETR-S "
            "tracker=motion-predictor nvtracker=disabled "
            "path=RTSP-NVDEC-mux-tiler-OSD-EGL",
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
