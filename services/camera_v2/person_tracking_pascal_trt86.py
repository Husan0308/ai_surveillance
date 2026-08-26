from __future__ import annotations

import os
import time
from pathlib import Path

from .person_tracking_trt86_audited import CameraPersonTrackingTRT86Audited

RESTART_EXIT_CODE = 75


def _set_key(lines: list[str], key: str, value: str, required: bool = True) -> bool:
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith(key + ":"):
            continue
        indent = line[: len(line) - len(stripped)]
        comment = ""
        if "#" in stripped:
            comment = "  #" + stripped.split("#", 1)[1]
        lines[index] = f"{indent}{key}: {value}{comment}"
        return True
    if required:
        raise RuntimeError(f"Pascal NvDCF config missing required key: {key}")
    return False


def _insert_target_management_key(lines: list[str], key: str, value: str) -> None:
    # DeepStream tracker YAML uses top-level section headers ending in ':'.
    start = None
    end = len(lines)
    for i, line in enumerate(lines):
        if line and not line[0].isspace() and line.split("#", 1)[0].strip() == "TargetManagement:":
            start = i
            break
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(["TargetManagement:", f"  {key}: {value}"])
        return
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line and not line[0].isspace() and line.split("#", 1)[0].strip().endswith(":"):
            end = i
            break
    for i in range(start + 1, end):
        stripped = lines[i].lstrip()
        if stripped.startswith(key + ":"):
            indent = lines[i][: len(lines[i]) - len(stripped)] or "  "
            lines[i] = f"{indent}{key}: {value}"
            return
    lines.insert(end, f"  {key}: {value}")


class CameraPersonTrackingPascalTRT86(CameraPersonTrackingTRT86Audited):
    """GTX 1050 Ti / Pascal-safe production baseline.

    Main process: DeepStream RTSP/NVDEC -> mux -> injected detector meta -> NvDCF
    -> tiler/OSD/EGL. Detector process: TensorRT 8.6.1 B1 engine via SHM.

    No Gst-nvinfer is used: DeepStream 7.1 ships TensorRT 10.x, which cannot build
    for SM 6.1 Pascal. Keeping TensorRT 8.6 in a separate process also avoids ABI
    collision with the DeepStream process.
    """

    def __init__(self) -> None:
        self._restart_requested = False
        self._restart_reason = ""
        self._source_started_at: dict[str, float] = {}
        self._last_frames: dict[str, int] = {}
        self._last_progress: dict[str, float] = {}
        super().__init__()
        self._stall_s = max(8.0, float(os.environ.get("CAMERA_V2_PASCAL_STALL_SEC", "12")))
        now = time.monotonic()
        self._last_frames = {cid: int(self.stats[cid].frames) for cid in self.sources}
        self._last_progress = {cid: now for cid in self.sources}
        print(
            "CAMERA_PASCAL_ARCH "
            "deepstream=decode/mux/NvDCF/OSD detector=TRT8.6-sidecar "
            "nvinfer=0 trt10=0 per-source-recycle=0 process-watchdog=1",
            flush=True,
        )

    @staticmethod
    def _stabilize_tracker_config(path: Path) -> Path:
        lines = path.read_text(encoding="utf-8").splitlines()
        _set_key(lines, "enableBboxUnClipping", "0")
        _set_key(lines, "minIouDiff4NewTarget", "0.72")
        _set_key(lines, "minTrackerConfidence", "0.12")
        _set_key(lines, "probationAge", "0")
        _set_key(lines, "maxShadowTrackingAge", "80")
        _set_key(lines, "earlyTerminationAge", "6")
        _set_key(lines, "minIou4TargetDuplicate", "0.94", required=False)
        _set_key(lines, "targetDuplicateRunInterval", "5", required=False)
        _insert_target_management_key(lines, "outputShadowTracks", "1")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        generated = path.read_text(encoding="utf-8")
        required = (
            "probationAge: 0",
            "maxShadowTrackingAge: 80",
            "earlyTerminationAge: 6",
            "outputShadowTracks: 1",
        )
        missing = [item for item in required if item not in generated]
        if missing:
            raise RuntimeError("Pascal NvDCF verification failed: " + ", ".join(missing))
        print(
            "CAMERA_PASCAL_NVDCF probationAge=0 maxShadowTrackingAge=80 "
            "earlyTerminationAge=6 outputShadowTracks=1 verified=1",
            flush=True,
        )
        return path

    def _add_camera(self, index, camera) -> None:
        super()._add_camera(index, camera)
        source = self.sources[camera.camera_id]
        # Bound nvurisrcbin's internal loop. If a source wedges after these tries,
        # the process watchdog recreates the complete graph cleanly.
        self._set_if(source, "rtsp-reconnect-interval", 2)
        self._set_if(source, "rtsp-reconnect-attempts", 3)
        self._set_if(source, "async-handling", True)

    def _startup_stagger_seconds(self) -> float:
        configured = float(getattr(self.settings.deepstream, "startup_stagger_sec", 0.5))
        return max(0.10, min(3.0, float(os.environ.get("CAMERA_V2_STARTUP_STAGGER_SEC", str(configured)))))

    def _prepare_staggered_sources(self) -> None:
        ordered = [camera.camera_id for camera in self.cameras]
        stagger = self._startup_stagger_seconds()
        for cid in ordered:
            source = self.sources[cid]
            source.set_locked_state(True)
            source.set_state(self.Gst.State.NULL)
        print(f"CAMERA_PASCAL_SOURCE_STAGGER order={ordered} interval={stagger:.2f}s", flush=True)

        for index, cid in enumerate(ordered):
            delay_ms = max(1, int(round(index * stagger * 1000.0)))

            def _start(camera_id=cid, ordinal=index):
                if self._stopping:
                    return False
                source = self.sources[camera_id]
                source.set_locked_state(False)
                sync = bool(source.sync_state_with_parent())
                now = time.monotonic()
                self._source_started_at[camera_id] = now
                self._last_progress[camera_id] = now
                self._last_frames[camera_id] = int(self.stats[camera_id].frames)
                print(f"CAMERA_PASCAL_SOURCE_START cid={camera_id} index={ordinal} sync={int(sync)}", flush=True)
                return False

            self.GLib.timeout_add(delay_ms, _start)

    def _source_watchdog(self) -> bool:
        if self._stopping:
            return False
        now = time.monotonic()
        for cid, started_at in list(self._source_started_at.items()):
            current = int(self.stats[cid].frames)
            if current != self._last_frames[cid]:
                self._last_frames[cid] = current
                self._last_progress[cid] = now
                continue
            if now - started_at < self._stall_s:
                continue
            stalled = now - self._last_progress[cid]
            if stalled < self._stall_s:
                continue
            self._restart_requested = True
            self._restart_reason = f"{cid} no-frames {stalled:.1f}s"
            print(f"CAMERA_PASCAL_PROCESS_RESTART reason={self._restart_reason} exit_code={RESTART_EXIT_CODE}", flush=True)
            self.stop()
            return False
        return True

    def run(self) -> int:
        self._prepare_staggered_sources()
        self.GLib.timeout_add_seconds(1, self._source_watchdog)
        result = super().run()
        if self._restart_requested:
            return RESTART_EXIT_CODE
        return result


def main() -> int:
    return CameraPersonTrackingPascalTRT86().run()


if __name__ == "__main__":
    raise SystemExit(main())
