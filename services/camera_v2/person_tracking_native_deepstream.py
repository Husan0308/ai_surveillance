from __future__ import annotations

import os
import time
from pathlib import Path

from .native_bridge import NativeMetaBridge
from .secure import SecureCameraWallV2

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / ".runtime" / "camera_v2"
MODEL_DIR = ROOT / "artifacts" / "yolo26s_deepstream"
ONNX_PATH = MODEL_DIR / "yolo26s-672x384-b6-e2e.onnx"
ENGINE_PATH = RUNTIME_DIR / "yolo26s-672x384-b6-fp16-deepstream.engine"
PARSER_PATH = (
    ROOT
    / "services"
    / "camera_v2"
    / "native_yolo26"
    / "libnvdsinfer_custom_yolo26_e2e.so"
)
PGIE_CONFIG = RUNTIME_DIR / "config_infer_primary_yolo26s_native.txt"
TRACKER_CONFIG = RUNTIME_DIR / "config_tracker_NvDCF_native.yml"
RESTART_EXIT_CODE = 75


def _deepstream_roots() -> list[Path]:
    roots = [Path("/opt/nvidia/deepstream/deepstream")]
    roots.extend(sorted(Path("/opt/nvidia/deepstream").glob("deepstream-*"), reverse=True))
    output: list[Path] = []
    seen: set[str] = set()
    for path in roots:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        output.append(path)
    return output


def _find_tracker_files() -> tuple[Path, Path]:
    library = None
    config = None
    for root in _deepstream_roots():
        candidate_lib = root / "lib/libnvds_nvmultiobjecttracker.so"
        candidate_cfg = root / "samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
        if library is None and candidate_lib.exists():
            library = candidate_lib
        if config is None and candidate_cfg.exists():
            config = candidate_cfg
    if library is None:
        raise RuntimeError("DeepStream NvMultiObjectTracker library was not found")
    if config is None:
        raise RuntimeError("DeepStream config_tracker_NvDCF_perf.yml was not found")
    return library, config


def _rewrite_key(lines: list[str], key: str, value: str, required: bool = True) -> None:
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
    if required:
        raise RuntimeError(f"NvDCF perf config missing required key: {key}")


def _native_tracker_config(stock: Path) -> Path:
    lines = stock.read_text(encoding="utf-8").splitlines()
    # Keep the stock DS7.1 perf profile and change only lifecycle parameters that
    # are necessary for sparse PGIE refresh. Do not carry the old experimental
    # ReID/pose-specific tuning into this baseline.
    _rewrite_key(lines, "minDetectorConfidence", "0.10")
    _rewrite_key(lines, "minTrackerConfidence", "0.12")
    _rewrite_key(lines, "probationAge", "0")
    _rewrite_key(lines, "maxShadowTrackingAge", "80")
    _rewrite_key(lines, "earlyTerminationAge", "6")
    _rewrite_key(lines, "outputShadowTracks", "1")
    _rewrite_key(lines, "enableReAssoc", "0", required=False)
    _rewrite_key(lines, "reidType", "0", required=False)
    _rewrite_key(lines, "outputReidTensor", "0", required=False)

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    TRACKER_CONFIG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return TRACKER_CONFIG


def _native_pgie_config(batch_size: int, gpu_id: int) -> Path:
    if not ONNX_PATH.is_file():
        raise RuntimeError(
            f"YOLO26 ONNX missing: {ONNX_PATH}. Run scripts/export_yolo26s_deepstream_onnx.py"
        )
    if not PARSER_PATH.is_file():
        raise RuntimeError(
            f"YOLO26 DeepStream parser missing: {PARSER_PATH}. Build services/camera_v2/native_yolo26 first"
        )

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    text = f"""[property]
gpu-id={gpu_id}
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file={ONNX_PATH}
model-engine-file={ENGINE_PATH}
batch-size={batch_size}
network-mode=2
num-detected-classes=80
interval=19
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=4
maintain-aspect-ratio=1
symmetric-padding=1
parse-bbox-func-name=NvDsInferParseCustomYolo26E2E
custom-lib-path={PARSER_PATH}

[class-attrs-all]
pre-cluster-threshold=0.10
post-cluster-threshold=0.10
"""
    PGIE_CONFIG.write_text(text, encoding="utf-8")
    return PGIE_CONFIG


class CameraPersonTrackingNativeDeepStream(SecureCameraWallV2):
    """Canonical DeepStream analytics path with no appsink/NumPy/PyTorch detector.

    RTSP/NVDEC -> nvstreammux -> nvinfer(YOLO26 TRT) -> NvDCF -> tiler -> OSD -> sink.
    """

    def __init__(self) -> None:
        self._native_restart_requested = False
        self._native_restart_reason = ""
        self._source_started_at: dict[str, float] = {}
        self._watchdog_last_frames: dict[str, int] = {}
        self._watchdog_last_progress: dict[str, float] = {}
        self._tracked_now = 0
        self._source_track_counts: dict[int, int] = {}
        self._tracker_batches = 0

        super().__init__()

        if self.Gst.ElementFactory.find("nvinfer") is None:
            raise RuntimeError("DeepStream Gst-nvinfer plugin is missing")
        if self.Gst.ElementFactory.find("nvtracker") is None:
            raise RuntimeError("DeepStream Gst-nvtracker plugin is missing")
        if len(self.cameras) != 6:
            raise RuntimeError("Native PGIE profile requires exactly six cameras")

        tracker_lib, stock_tracker = _find_tracker_files()
        tracker_config = _native_tracker_config(stock_tracker)
        pgie_config = _native_pgie_config(len(self.cameras), self.gpu_id)

        self.bridge = NativeMetaBridge()
        self._install_native_analytics(pgie_config, tracker_lib, tracker_config)

        now = time.monotonic()
        self._watchdog_last_frames = {cid: 0 for cid in self.sources}
        self._watchdog_last_progress = {cid: now for cid in self.sources}
        self._watchdog_stall_s = max(
            8.0, float(os.environ.get("CAMERA_V2_NATIVE_STALL_SEC", "12.0"))
        )

        print(
            "CAMERA_NATIVE_PIPELINE "
            "topology=nvurisrcbin/NVDEC->nvstreammux->nvinfer(YOLO26)->nvtracker(NvDCF)->"
            "nvmultistreamtiler->nvvideoconvert->nvdsosd->nveglglessink "
            f"mux={self.frame_width}x{self.frame_height}/batch{len(self.cameras)} "
            "pgie=672x384/fp16/interval19 tracker=640x384 appsink=0 numpy=0 pytorch=0",
            flush=True,
        )

    def _configure_queue(self, queue) -> None:
        self._set_if(queue, "max-size-buffers", 4)
        self._set_if(queue, "max-size-bytes", 0)
        self._set_if(queue, "max-size-time", 0)
        self._set_if(queue, "leaky", 0)
        self._set_if(queue, "silent", True)

    def _install_native_analytics(
        self, pgie_config: Path, tracker_lib: Path, tracker_config: Path
    ) -> None:
        if not self.mux.unlink(self.tiler):
            raise RuntimeError("Could not detach nvstreammux -> tiler")

        mux_queue = self._make("queue", "native_mux_pgie_queue")
        pgie = self._make("nvinfer", "native_yolo26_pgie")
        track_queue = self._make("queue", "native_pgie_tracker_queue")
        tracker = self._make("nvtracker", "native_nvdcf_tracker")

        self._configure_queue(mux_queue)
        self._configure_queue(track_queue)
        pgie.set_property("config-file-path", str(pgie_config))
        self._set_if(pgie, "batch-size", len(self.cameras))
        self._set_if(pgie, "interval", 19)
        self._set_if(pgie, "gpu-id", self.gpu_id)

        self._set_if(tracker, "tracker-width", 640)
        self._set_if(tracker, "tracker-height", 384)
        tracker.set_property("ll-lib-file", str(tracker_lib))
        tracker.set_property("ll-config-file", str(tracker_config))
        self._set_if(tracker, "gpu-id", self.gpu_id)
        self._set_if(tracker, "compute-hw", 1)
        self._set_if(tracker, "enable-batch-process", True)
        self._set_if(tracker, "display-tracking-id", False)
        self._set_if(tracker, "tracking-id-reset-mode", 1)

        for element in (mux_queue, pgie, track_queue, tracker):
            self.pipeline.add(element)

        chain = [self.mux, mux_queue, pgie, track_queue, tracker, self.tiler]
        for src, dst in zip(chain, chain[1:]):
            if not src.link(dst):
                raise RuntimeError(
                    f"Native DeepStream link failed: {src.get_name()} -> {dst.get_name()}"
                )

        if not self.wall_queue.unlink(self.sink):
            raise RuntimeError("Could not detach wall queue -> sink")
        convert = self._make("nvvideoconvert", "native_wall_convert")
        caps = self._make("capsfilter", "native_wall_rgba")
        osd = self._make("nvdsosd", "native_wall_osd")
        self._set_if(convert, "gpu-id", self.gpu_id)
        caps.set_property(
            "caps", self.Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
        )
        self._set_if(osd, "process-mode", 1)
        self._set_if(osd, "display-bbox", True)
        self._set_if(osd, "display-text", False)
        self._set_if(osd, "display-mask", False)
        self._set_if(osd, "gpu-id", self.gpu_id)
        for element in (convert, caps, osd):
            self.pipeline.add(element)
        for src, dst in zip(
            [self.wall_queue, convert, caps, osd], [convert, caps, osd, self.sink]
        ):
            if not src.link(dst):
                raise RuntimeError(
                    f"Native display link failed: {src.get_name()} -> {dst.get_name()}"
                )

        tracker.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER, self._native_tracker_probe
        )
        self.pgie = pgie
        self.tracker = tracker
        self.osd = osd

    def _native_tracker_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        try:
            rows = self.bridge.copy_tracks(buffer, max_rows=256)
            ids_by_source: dict[int, set[int]] = {
                index: set() for index in range(len(self.cameras))
            }
            for row in rows:
                source_id = int(row.get("source_id", -1))
                object_id = int(row.get("object_id", -1))
                if source_id in ids_by_source and object_id >= 0:
                    ids_by_source[source_id].add(object_id)
            self._source_track_counts = {
                source_id: len(object_ids)
                for source_id, object_ids in ids_by_source.items()
            }
            self._tracked_now = sum(self._source_track_counts.values())
            self._tracker_batches += 1
            self.bridge.apply_local_track_style(buffer)
        except Exception as exc:
            print(
                f"CAMERA_NATIVE_TRACK warning={type(exc).__name__}:{exc}", flush=True
            )
        return self.Gst.PadProbeReturn.OK

    def _startup_stagger_seconds(self) -> float:
        configured = float(getattr(self.settings.deepstream, "startup_stagger_sec", 0.5))
        return max(
            0.10,
            min(
                3.0,
                float(os.environ.get("CAMERA_V2_STARTUP_STAGGER_SEC", str(configured))),
            ),
        )

    def _schedule_sources(self) -> None:
        ordered = [camera.camera_id for camera in self.cameras]
        stagger = self._startup_stagger_seconds()
        for cid in ordered:
            source = self.sources[cid]
            source.set_locked_state(True)
            source.set_state(self.Gst.State.NULL)

        print(
            f"CAMERA_NATIVE_SOURCE_STAGGER order={ordered} interval={stagger:.2f}s",
            flush=True,
        )

        for index, cid in enumerate(ordered):
            delay_ms = max(1, int(round(index * stagger * 1000.0)))

            def _start(camera_id=cid, ordinal=index):
                if self._stopping:
                    return False
                source = self.sources[camera_id]
                source.set_locked_state(False)
                ok = bool(source.sync_state_with_parent())
                now = time.monotonic()
                self._source_started_at[camera_id] = now
                self._watchdog_last_progress[camera_id] = now
                self._watchdog_last_frames[camera_id] = int(
                    self.stats[camera_id].frames
                )
                print(
                    f"CAMERA_NATIVE_SOURCE_START cid={camera_id} index={ordinal} sync={int(ok)}",
                    flush=True,
                )
                return False

            self.GLib.timeout_add(delay_ms, _start)

    def _source_watchdog(self) -> bool:
        if self._stopping:
            return False
        now = time.monotonic()
        for cid, started_at in list(self._source_started_at.items()):
            current = int(self.stats[cid].frames)
            if current != self._watchdog_last_frames[cid]:
                self._watchdog_last_frames[cid] = current
                self._watchdog_last_progress[cid] = now
                continue
            if now - started_at < self._watchdog_stall_s:
                continue
            stalled = now - self._watchdog_last_progress[cid]
            if stalled < self._watchdog_stall_s:
                continue

            # Do NOT NULL->PLAYING an existing nvurisrcbin here. DeepStream owns
            # internal ghost pads; recycling that same bin can create duplicate
            # vsrc_0 pads and a not-linked RTSP loop. Restart the whole graph cleanly.
            self._native_restart_requested = True
            self._native_restart_reason = f"{cid} no-frames {stalled:.1f}s"
            print(
                "CAMERA_NATIVE_PROCESS_RESTART "
                f"reason={self._native_restart_reason} exit_code={RESTART_EXIT_CODE}",
                flush=True,
            )
            self.stop()
            return False
        return True

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        print(
            "CAMERA_NATIVE_TRACK "
            f"tracked_now={self._tracked_now} source_counts={self._source_track_counts} "
            f"tracker_batches={self._tracker_batches} pgie_interval=19 appsink=0",
            flush=True,
        )
        return keep

    def run(self) -> int:
        self._schedule_sources()
        self.GLib.timeout_add_seconds(1, self._source_watchdog)
        result = super().run()
        if self._native_restart_requested:
            return RESTART_EXIT_CODE
        return result


def main() -> int:
    return CameraPersonTrackingNativeDeepStream().run()


if __name__ == "__main__":
    raise SystemExit(main())
