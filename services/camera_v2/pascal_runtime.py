from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from .detection import INFER_HEIGHT, INFER_WIDTH
from .person_tracking_pascal_trt86 import CameraPersonTrackingPascalTRT86
from .person_tracking_trt86_fresh import CameraPersonTrackingTRT86Fresh


def _replace_yaml_key(lines: list[str], key: str, value: str) -> None:
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith(key + ":"):
            continue
        indent = line[: len(line) - len(stripped)]
        comment = ""
        if "#" in stripped:
            comment = "  #" + stripped.split("#", 1)[1]
        lines[index] = f"{indent}{key}: {value}{comment}"
        return
    raise RuntimeError(f"Pascal continuity config missing required NvDCF key: {key}")


class CameraPascalRuntime(CameraPersonTrackingPascalTRT86):
    """GTX 1050 Ti runtime with sharp display and isolated sparse TRT8.6 analytics.

    The presentation and detector paths intentionally have different quality/
    geometry policies. The live wall keeps enough pixels for a clear 3x2 view;
    the detector branch only wakes for requested inference frames and therefore
    can use cubic scaling without taxing every decoded frame.
    """

    @staticmethod
    def _stabilize_tracker_config(path: Path) -> Path:
        # At the guarded detector cadence fresh observations remain comfortably
        # inside a seven-second NvDCF shadow lifetime. NvDCF owns per-frame motion;
        # YOLO refreshes object evidence instead of being used as a video clock.
        path = CameraPersonTrackingPascalTRT86._stabilize_tracker_config(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        _replace_yaml_key(lines, "minIouDiff4NewTarget", "0.45")
        _replace_yaml_key(lines, "minTrackerConfidence", "0.05")
        _replace_yaml_key(lines, "maxShadowTrackingAge", "140")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        final = path.read_text(encoding="utf-8")
        required = (
            "minIouDiff4NewTarget: 0.45",
            "minTrackerConfidence: 0.05",
            "probationAge: 0",
            "maxShadowTrackingAge: 140",
            "outputShadowTracks: 1",
        )
        missing = [item for item in required if item not in final]
        if missing:
            raise RuntimeError("Pascal continuity verification failed: " + ", ".join(missing))
        print(
            "CAMERA_PASCAL_CONTINUITY "
            "minIouDiff4NewTarget=0.45 minTrackerConfidence=0.05 "
            "probationAge=0 maxShadowTrackingAge=140 outputShadowTracks=1 verified=1",
            flush=True,
        )
        return path

    def _configure_mux(self) -> None:
        super()._configure_mux()
        # NVIDIA dGPU interpolation method 2 is cubic. The previous latency patch
        # changed this to bilinear and visibly softened the already-downscaled wall.
        # Cubic is the quality/perf compromise for six input streams; the final
        # presentation downscale below uses Lanczos.
        self._set_if(self.mux, "interpolation-method", 2)
        self._set_if(self.mux, "compute-hw", 1)
        self._set_if(self.mux, "buffer-pool-size", 12)

    def _configure_tiler(self) -> None:
        super()._configure_tiler()
        # The tiler is the last geometric downscale before OSD/EGL. Use dGPU
        # Lanczos (method 4) here so 640x360 tiles do not look washed/soft.
        self._set_if(self.tiler, "interpolation-method", 4)
        self._set_if(self.tiler, "compute-hw", 1)

    def _configure_rtsp_child(self, _bin, _sub_bin, element, camera) -> None:
        super()._configure_rtsp_child(_bin, _sub_bin, element, camera)
        factory = element.get_factory()
        factory_name = factory.get_name() if factory is not None else ""
        if factory_name != "rtspsrc":
            return
        if self._transport() == "tcp":
            # Keep the useful latency fix: receive-time timestamping prevents TCP
            # RTP clock drift from silently accumulating end-to-end delay.
            self._set_if(element, "tcp-timestamp", True)
            print(
                f"CAMERA_PASCAL_RTSP {camera.camera_id} tcp_timestamp=1 "
                f"latency={self.rtsp_latency_ms}ms drop_on_latency=1",
                flush=True,
            )

    def _add_camera(self, index, camera) -> None:
        """Build a true 16:9 detector capture path without nvvideoconvert dest-crop.

        The audited parent used dest-crop=0:3:672:378 while negotiating a 672x384
        output surface. On the real 2560x1440 streams nvvideoconvert reports
        `Cannot keep DAR`. Instead, ask GStreamer for an exact 672x378 16:9 BGRx
        frame, then add the required 3+3 rows of value 114 in host memory. Only a
        requested sparse frame reaches this converter because Fresh keeps the gate
        directly in front of it.
        """
        # Skip CameraPersonTrackingTRT86Audited._add_camera because that is exactly
        # where the dest-crop workaround is installed. Keep all Fresh JIT capture
        # behavior and reproduce the Pascal source hardening below.
        CameraPersonTrackingTRT86Fresh._add_camera(self, index, camera)
        cid = camera.camera_id

        source = self.pipeline.get_by_name(f"camera_v2_source_{index}")
        converter = self.pipeline.get_by_name(f"detect_convert_{index}")
        capsfilter = self.pipeline.get_by_name(f"detect_caps_{index}")
        if source is None or converter is None or capsfilter is None:
            raise RuntimeError(f"{cid}: Pascal detector branch incomplete")

        transport = self._transport()
        self._set_if(source, "select-rtp-protocol", 4 if transport == "tcp" else 0)
        self._set_if(source, "rtsp-reconnect-interval", 2)
        self._set_if(source, "rtsp-reconnect-attempts", 3)
        self._set_if(source, "async-handling", True)

        content_h = int(round(INFER_WIDTH * 9.0 / 16.0))
        if content_h != 378 or INFER_HEIGHT != 384:
            raise RuntimeError(
                f"unexpected TRT geometry tensor={INFER_WIDTH}x{INFER_HEIGHT} "
                f"content_h={content_h}"
            )
        capsfilter.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,width={INFER_WIDTH},height={content_h},"
                "pixel-aspect-ratio=1/1"
            ),
        )
        self._set_if(converter, "interpolation-method", 2)
        self._set_if(converter, "compute-hw", 1)
        self._detector_letterbox = (0, 3, INFER_WIDTH, content_h)

        if cid == "CAM-01":
            print(
                "CAM01_TRT86_SOURCE_HARDENED "
                f"outer_transport={transport} async_handling=1",
                flush=True,
            )
            print(
                "CAM01_TRT86_PREPROCESS "
                f"tensor={INFER_WIDTH}x{INFER_HEIGHT} "
                f"content={INFER_WIDTH}x{content_h} pad=0,3 "
                "mode=exact-16:9-caps+host-pad114 interpolation=cubic",
                flush=True,
            )

    def _on_infer_sample(self, sink, cid: str):
        """Pack an exact 672x378 capture into a 672x384 YOLO letterbox tensor."""
        sample = sink.emit("pull-sample")
        if sample is None:
            with self.capture_lock:
                self.capture_requested[cid] = True
            return self.Gst.FlowReturn.OK

        try:
            structure = sample.get_caps().get_structure(0)
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))
            expected_h = 378
            if width != INFER_WIDTH or height != expected_h:
                raise RuntimeError(
                    f"{cid}: detector capture={width}x{height}, "
                    f"expected={INFER_WIDTH}x{expected_h}"
                )

            buffer = sample.get_buffer()
            ok, mapped = buffer.map(self.Gst.MapFlags.READ)
            if not ok:
                raise RuntimeError(f"{cid}: detector BGRx map failed")
            try:
                tight_stride = width * 4
                mapped_size = int(getattr(mapped, "size", len(mapped.data)))
                if mapped_size < tight_stride * height:
                    raise RuntimeError(
                        f"{cid}: BGRx buffer too small: "
                        f"{mapped_size} < {tight_stride * height}"
                    )
                row_stride = (
                    mapped_size // height
                    if mapped_size % height == 0
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

                # Fixed TensorRT B1 input. 672x378 is exact 16:9; only the 3-row
                # bars are synthetic. There is no geometric crop or stretch.
                frame = np.full(
                    (INFER_HEIGHT, INFER_WIDTH, 3),
                    114,
                    dtype=np.uint8,
                )
                frame[3:381, :, :] = bgrx[..., :3]
            finally:
                buffer.unmap(mapped)

            now = time.monotonic()
            self.mailbox.put(cid, now, frame)

            if cid not in self._infer_stride_logged:
                self._infer_stride_logged.add(cid)
                print(
                    f"CAMERA_INFER_LAYOUT {cid} capture={width}x{height} "
                    f"stride={row_stride} tensor={INFER_WIDTH}x{INFER_HEIGHT} "
                    "letterbox=3+378+3 pad114",
                    flush=True,
                )
            if cid not in self._capture_sample_logged:
                self._capture_sample_logged.add(cid)
                print(
                    f"CAM01_TRT86_CAPTURE_SAMPLE camera={cid} first_sample=1",
                    flush=True,
                )
            return self.Gst.FlowReturn.OK
        except Exception as exc:
            with self.capture_lock:
                self.capture_requested[cid] = True
            print(
                f"CAMERA_PASCAL_INFER_CAPTURE {cid} "
                f"warning={type(exc).__name__}:{exc}",
                file=sys.stderr,
                flush=True,
            )
            return self.Gst.FlowReturn.OK

    def __init__(self) -> None:
        super().__init__()
        # Rectangle-only OSD keeps the tracker display cheap. Text can be restored
        # later without touching the camera geometry.
        self._set_if(self.osd, "display-text", False)
        self._set_if(self.osd, "display-bbox", True)

        mux_interp = (
            self.mux.get_property("interpolation-method")
            if self.mux.find_property("interpolation-method")
            else "n/a"
        )
        tiler_interp = (
            self.tiler.get_property("interpolation-method")
            if self.tiler.find_property("interpolation-method")
            else "n/a"
        )
        pool = (
            self.mux.get_property("buffer-pool-size")
            if self.mux.find_property("buffer-pool-size")
            else "n/a"
        )
        print(
            "CAMERA_PASCAL_QUALITY "
            f"mux={self.frame_width}x{self.frame_height}/cubic "
            f"tiler={self.wall_width}x{self.wall_height}/lanczos "
            f"tracker={self.tracker_width}x{self.tracker_height} "
            f"mux_interp={mux_interp} tiler_interp={tiler_interp} pool={pool} "
            "detector=672x378/cubic+pad114 display_detector_independent=1 "
            "osd_text=0 latest_queues=1",
            flush=True,
        )

    def _print_stats(self) -> bool:
        # Do not let a tiled-wall render metric throttle YOLO. NvDCF is per-frame;
        # detector cadence is an explicit GPU-budget choice, not a presentation-FPS
        # feedback loop.
        with self.det_lock:
            target_hz = float(self.detector_target_hz)
            saved_min_hz = float(self.detector_min_hz)
            saved_max_hz = float(self.detector_max_hz)
            self.detector_min_hz = target_hz
            self.detector_max_hz = target_hz
        try:
            keep = super()._print_stats()
        finally:
            with self.det_lock:
                self.detector_target_hz = target_hz
                self.detector_min_hz = saved_min_hz
                self.detector_max_hz = saved_max_hz
        print(
            "CAMERA_PASCAL_RATE_GUARD "
            f"target_hz={target_hz:.2f}/cam wall_feedback=disabled "
            "reason=explicit_pascal_gpu_budget",
            flush=True,
        )
        return keep

    def _source_to_tee(self, _source, pad, tee, cid: str) -> None:
        # Dynamic pads can appear before fixed caps. Audio is disabled, so unknown
        # caps may be linked; known non-video pads are rejected.
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
        if sink is None:
            print(
                f"CAMERA_PASCAL {cid} tee sink pad missing",
                file=sys.stderr,
                flush=True,
            )
            return
        if sink.is_linked():
            return
        result = pad.link(sink)
        if result != self.Gst.PadLinkReturn.OK:
            caps_text = caps.to_string() if caps is not None else "pending"
            print(
                f"CAMERA_PASCAL {cid} source->tee link failed "
                f"result={result} caps={caps_text}",
                file=sys.stderr,
                flush=True,
            )
            return
        caps_text = caps.to_string() if caps is not None else "pending"
        print(
            f"CAMERA_PASCAL {cid} source->tee linked caps={caps_text}",
            flush=True,
        )


def main() -> int:
    return CameraPascalRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
