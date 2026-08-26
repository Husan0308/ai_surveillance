from __future__ import annotations

import sys
from pathlib import Path

from .person_tracking_pascal_trt86 import CameraPersonTrackingPascalTRT86


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
    """Final GTX 1050 Ti runtime: smooth wall + Pascal-safe TRT8.6 analytics."""

    @staticmethod
    def _stabilize_tracker_config(path: Path) -> Path:
        # First apply the Pascal-safe base profile, then add the continuity tuning
        # justified by the live 20 FPS measurements. With a guarded ~0.5 detector
        # Hz, fresh detector observations arrive about every two seconds per camera.
        # 140 shadow frames (~7 s at 20 FPS) then survives two missed refreshes
        # without keeping stale targets alive indefinitely.
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
        # Pascal has little spare GPU after TRT8.6 + NvDCF. Lanczos (4) was
        # needlessly expensive for 6 live CCTV streams. Bilinear (1) keeps the
        # wall clear while materially reducing the scaling workload.
        self._set_if(self.mux, "interpolation-method", 1)
        self._set_if(self.mux, "compute-hw", 1)
        self._set_if(self.mux, "buffer-pool-size", 12)

    def _configure_tiler(self) -> None:
        super()._configure_tiler()
        self._set_if(self.tiler, "interpolation-method", 1)
        self._set_if(self.tiler, "compute-hw", 1)

    def _configure_rtsp_child(self, _bin, _sub_bin, element, camera) -> None:
        super()._configure_rtsp_child(_bin, _sub_bin, element, camera)
        factory = element.get_factory()
        factory_name = factory.get_name() if factory is not None else ""
        if factory_name != "rtspsrc":
            return
        if self._transport() == "tcp":
            # GStreamer documents that TCP RTP timestamps can drift against the
            # client clock and make observed end-to-end latency grow over time.
            # Timestamping each received TCP packet prevents that accumulation.
            self._set_if(element, "tcp-timestamp", True)
            print(
                f"CAMERA_PASCAL_RTSP {camera.camera_id} tcp_timestamp=1 "
                f"latency={self.rtsp_latency_ms}ms drop_on_latency=1",
                flush=True,
            )

    def __init__(self) -> None:
        super().__init__()
        # Text rendering is not required for the sticky-bbox baseline and costs
        # extra OSD work. Keep the rectangle itself on the GPU OSD path.
        self._set_if(self.osd, "display-text", False)
        self._set_if(self.osd, "display-bbox", True)

        # The detector branch previously inherited cubic scaling from the generic
        # runtime. On this GPU the detector result itself is the expensive path;
        # bilinear scaling is sufficient for 672x384 person detection and leaves
        # more GPU time for TensorRT + NvDCF.
        for index in range(len(self.cameras)):
            converter = self.pipeline.get_by_name(f"detect_convert_{index}")
            if converter is not None:
                self._set_if(converter, "interpolation-method", 1)
                self._set_if(converter, "compute-hw", 1)

        mux_interp = self.mux.get_property("interpolation-method") if self.mux.find_property("interpolation-method") else "n/a"
        tiler_interp = self.tiler.get_property("interpolation-method") if self.tiler.find_property("interpolation-method") else "n/a"
        pool = self.mux.get_property("buffer-pool-size") if self.mux.find_property("buffer-pool-size") else "n/a"
        print(
            "CAMERA_PASCAL_SMOOTHNESS "
            f"mux={self.frame_width}x{self.frame_height}/bilinear "
            f"tiler={self.wall_width}x{self.wall_height}/bilinear "
            f"tracker={self.tracker_width}x{self.tracker_height} "
            f"mux_interp={mux_interp} tiler_interp={tiler_interp} pool={pool} "
            "detector_scale=bilinear osd_text=0 latest_queues=1",
            flush=True,
        )

    def _print_stats(self) -> bool:
        # Critical latency guard. CameraPersonTrackingFinal historically used the
        # tiled wall's frame-interval p95 as detector feedback. On this six-camera
        # Pascal graph wall p95 can be ~90-115 ms even while every source remains
        # healthy at ~20 FPS with q=0. That false signal repeatedly lowered the
        # detector target until it reached ~0.1 Hz/camera (one fresh detection per
        # ~10 seconds), which is longer than NvDCF's 7-second shadow lifetime.
        # Freeze the configured detector cadence while parent stats are collected.
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
            "reason=source_fps_is_authoritative",
            flush=True,
        )
        return keep

    def _source_to_tee(self, _source, pad, tee, cid: str) -> None:
        # pad-added can fire before fixed caps are available. Returning in that
        # state is unsafe because the dynamic pad is not guaranteed to be emitted
        # again when caps later become fixed. Audio is disabled on nvurisrcbin, so
        # unknown caps may be linked; known non-video caps are still rejected.
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
            print(f"CAMERA_PASCAL {cid} tee sink pad missing", file=sys.stderr, flush=True)
            return
        if sink.is_linked():
            return
        result = pad.link(sink)
        if result != self.Gst.PadLinkReturn.OK:
            caps_text = caps.to_string() if caps is not None else "pending"
            print(
                f"CAMERA_PASCAL {cid} source->tee link failed result={result} caps={caps_text}",
                file=sys.stderr,
                flush=True,
            )
            return
        caps_text = caps.to_string() if caps is not None else "pending"
        print(f"CAMERA_PASCAL {cid} source->tee linked caps={caps_text}", flush=True)


def main() -> int:
    return CameraPascalRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
