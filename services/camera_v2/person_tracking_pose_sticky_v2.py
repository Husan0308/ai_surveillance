from __future__ import annotations

"""Verified pose-sticky runtime for DeepStream 7.1.

CameraPersonTrackingFinal resolves the generated NvDCF YAML through
``self._stabilize_tracker_config``. This subclass therefore owns the final file
passed to nvtracker and verifies the parameters that are part of the current
DeepStream 7.1 NvMultiObjectTracker configuration contract.

The six RTSP sources start one-by-one so the NVR is not hit by six simultaneous
session creations. Runtime recovery is also source-aware: a frame-stall watchdog
first hard-recycles only the stalled nvurisrcbin. If the bin remains wedged after
bounded retries, the process exits with code 75 so the launcher can perform a
clean whole-pipeline restart while preserving staggered source startup.
"""

import os
import time
from pathlib import Path

from .person_tracking_pose_sticky import CameraPersonTrackingPoseSticky


_REQUIRED_EFFECTIVE: dict[str, str] = {
    "minDetectorConfidence": "0.03",
    "minTrackerConfidence": "0.08",
    "probationAge": "0",
    "maxShadowTrackingAge": "100",
    "earlyTerminationAge": "6",
    "tentativeDetectorConfidence": "0.03",
    "outputShadowTracks": "1",
}

_SECTION_FOR_MISSING: dict[str, str] = {
    "tentativeDetectorConfidence": "DataAssociator",
    "outputShadowTracks": "TargetManagement",
}

_LEGACY_OPTIONAL: dict[str, str] = {
    "minTrackingConfidenceDuringInactive": "0.05",
}

RESTART_EXIT_CODE = 75


def _section_header(line: str) -> str | None:
    if not line or line[0].isspace():
        return None
    body = line.split("#", 1)[0].strip()
    if body.endswith(":") and body[:-1].strip():
        return body[:-1].strip()
    return None


def _section_bounds(lines: list[str], section: str) -> tuple[int, int] | None:
    start = None
    for index, line in enumerate(lines):
        if _section_header(line) == section:
            start = index
            break
    if start is None:
        return None
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if _section_header(lines[index]) is not None:
            end = index
            break
    return start, end


def _ensure_section_key(lines: list[str], section: str, key: str, value: str) -> None:
    bounds = _section_bounds(lines, section)
    if bounds is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"{section}:")
        lines.append(f"  {key}: {value}")
        return

    start, end = bounds
    for index in range(start + 1, end):
        stripped = lines[index].lstrip()
        if stripped.startswith(key + ":"):
            indent = lines[index][: len(lines[index]) - len(stripped)] or "  "
            comment = ""
            if "#" in stripped:
                comment = "  #" + stripped.split("#", 1)[1]
            lines[index] = f"{indent}{key}: {value}{comment}"
            return

    lines.insert(end, f"  {key}: {value}")


def _rewrite_existing(lines: list[str], values: dict[str, str]) -> set[str]:
    seen: set[str] = set()
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        for key, value in values.items():
            if not stripped.startswith(key + ":"):
                continue
            comment = ""
            if "#" in stripped:
                comment = "  #" + stripped.split("#", 1)[1]
            lines[index] = f"{indent}{key}: {value}{comment}"
            seen.add(key)
            break
    return seen


def _parse_effective(path: Path, keys: set[str]) -> dict[str, str]:
    effective: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        body = line.split("#", 1)[0].strip()
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        key = key.strip()
        if key in keys:
            effective[key] = value.strip()
    return effective


def _rewrite_yaml_keys(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    seen = _rewrite_existing(lines, _REQUIRED_EFFECTIVE)

    for key, value in _REQUIRED_EFFECTIVE.items():
        if key in seen:
            continue
        section = _SECTION_FOR_MISSING.get(key)
        if section is None:
            raise RuntimeError(
                "pose NvDCF config is missing required DeepStream 7.1 key "
                f"{key!r} and no safe insertion section is defined"
            )
        _ensure_section_key(lines, section, key, value)

    legacy_seen = _rewrite_existing(lines, _LEGACY_OPTIONAL)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    effective = _parse_effective(path, set(_REQUIRED_EFFECTIVE))
    wrong = {
        key: (_REQUIRED_EFFECTIVE[key], effective.get(key))
        for key in _REQUIRED_EFFECTIVE
        if effective.get(key) != _REQUIRED_EFFECTIVE[key]
    }
    if wrong:
        raise RuntimeError(f"pose NvDCF effective config mismatch: {wrong}")

    legacy_effective = _parse_effective(path, set(_LEGACY_OPTIONAL))
    if legacy_seen:
        wrong_legacy = {
            key: (_LEGACY_OPTIONAL[key], legacy_effective.get(key))
            for key in legacy_seen
            if legacy_effective.get(key) != _LEGACY_OPTIONAL[key]
        }
        if wrong_legacy:
            raise RuntimeError(f"pose NvDCF legacy config mismatch: {wrong_legacy}")

    return effective, legacy_effective


class CameraPersonTrackingPoseStickyV2(CameraPersonTrackingPoseSticky):
    """Pose refresh + verified NvDCF + resilient multi-RTSP source lifecycle."""

    def _add_camera(self, index, camera) -> None:
        super()._add_camera(index, camera)
        source = self.sources.get(camera.camera_id)
        if source is None:
            raise RuntimeError(f"{camera.camera_id}: source missing after construction")

        # Do not allow nvurisrcbin to retry a dead socket forever. The app-level
        # watchdog below owns escalation when the bin's internal reconnect wedges.
        reconnect_attempts = max(
            1,
            min(
                20,
                int(os.environ.get("CAMERA_V2_RTSP_RECONNECT_ATTEMPTS", "3")),
            ),
        )
        self._set_if(source, "rtsp-reconnect-interval", 2)
        self._set_if(source, "rtsp-reconnect-attempts", reconnect_attempts)
        self._set_if(source, "init-rtsp-reconnect-interval", 2)
        print(
            "CAMERA_POSE_SOURCE_RECOVERY "
            f"cid={camera.camera_id} internal_reconnect=2s/{reconnect_attempts}",
            flush=True,
        )

    def __init__(self) -> None:
        super().__init__()
        now = time.monotonic()
        self._source_started_at: dict[str, float] = {}
        self._watchdog_last_frames = {
            camera.camera_id: int(self.stats[camera.camera_id].frames)
            for camera in self.cameras
        }
        self._watchdog_last_progress = {
            camera.camera_id: now for camera in self.cameras
        }
        self._source_recycle_count = {
            camera.camera_id: 0 for camera in self.cameras
        }
        self._source_recycling: set[str] = set()
        self._restart_requested = False
        self._restart_reason = ""
        self._watchdog_stall_s = max(
            3.0,
            float(os.environ.get("CAMERA_V2_SOURCE_WATCHDOG_STALL_SEC", "6.0")),
        )
        self._watchdog_grace_s = max(
            3.0,
            float(os.environ.get("CAMERA_V2_SOURCE_WATCHDOG_GRACE_SEC", "6.0")),
        )
        self._watchdog_recycle_pause_ms = max(
            200,
            int(os.environ.get("CAMERA_V2_SOURCE_RECYCLE_PAUSE_MS", "750")),
        )
        self._watchdog_max_recycles = max(
            1,
            min(
                5,
                int(os.environ.get("CAMERA_V2_SOURCE_MAX_RECYCLES", "2")),
            ),
        )
        print(
            "CAMERA_POSE_SOURCE_WATCHDOG "
            f"stall={self._watchdog_stall_s:.1f}s grace={self._watchdog_grace_s:.1f}s "
            f"recycle_pause={self._watchdog_recycle_pause_ms}ms "
            f"max_recycles={self._watchdog_max_recycles} process_fallback=1",
            flush=True,
        )

    @staticmethod
    def _stabilize_tracker_config(path: Path) -> Path:
        path = Path(path)
        effective, legacy = _rewrite_yaml_keys(path)
        print(
            "CAMERA_POSE_NVDCF_EFFECTIVE "
            + " ".join(f"{key}={effective[key]}" for key in _REQUIRED_EFFECTIVE)
            + " legacyInactiveConf="
            + (legacy.get("minTrackingConfidenceDuringInactive", "not-present-ds7.1")),
            flush=True,
        )
        return path

    def _startup_stagger_seconds(self) -> float:
        configured = float(getattr(self.settings.deepstream, "startup_stagger_sec", 0.5))
        return max(
            0.10,
            min(
                5.0,
                float(os.environ.get("CAMERA_V2_STARTUP_STAGGER_SEC", str(configured))),
            ),
        )

    def _schedule_staggered_sources(self) -> None:
        stagger_s = self._startup_stagger_seconds()
        ordered = [camera.camera_id for camera in self.cameras]

        for cid in ordered:
            source = self.sources.get(cid)
            if source is None:
                raise RuntimeError(f"{cid}: source missing before stagger startup")
            source.set_locked_state(True)
            source.set_state(self.Gst.State.NULL)

        print(
            "CAMERA_POSE_SOURCE_STAGGER "
            f"order={ordered} interval={stagger_s:.2f}s locked_at_null=1",
            flush=True,
        )

        for index, cid in enumerate(ordered):
            delay_ms = max(1, int(round(index * stagger_s * 1000.0)))

            def _start_one(camera_id=cid, ordinal=index):
                if self._stopping:
                    return False
                source = self.sources.get(camera_id)
                if source is None:
                    print(
                        f"CAMERA_POSE_SOURCE_START cid={camera_id} error=missing-source",
                        flush=True,
                    )
                    return False
                source.set_locked_state(False)
                ok = bool(source.sync_state_with_parent())
                now = time.monotonic()
                self._source_started_at[camera_id] = now
                self._watchdog_last_progress[camera_id] = now
                self._watchdog_last_frames[camera_id] = int(self.stats[camera_id].frames)
                print(
                    "CAMERA_POSE_SOURCE_START "
                    f"cid={camera_id} index={ordinal} sync={int(ok)}",
                    flush=True,
                )
                return False

            self.GLib.timeout_add(delay_ms, _start_one)

    def _hard_recycle_source(self, cid: str, stalled_s: float) -> None:
        source = self.sources.get(cid)
        if source is None or cid in self._source_recycling:
            return

        self._source_recycling.add(cid)
        attempt = self._source_recycle_count[cid]
        source.set_locked_state(True)
        state_result = source.set_state(self.Gst.State.NULL)
        print(
            "CAMERA_POSE_SOURCE_RECYCLE "
            f"cid={cid} attempt={attempt}/{self._watchdog_max_recycles} "
            f"stalled={stalled_s:.1f}s phase=NULL state={int(state_result)}",
            flush=True,
        )

        def _restart_one() -> bool:
            if self._stopping:
                self._source_recycling.discard(cid)
                return False
            current = self.sources.get(cid)
            if current is None:
                self._source_recycling.discard(cid)
                return False
            current.set_locked_state(False)
            ok = bool(current.sync_state_with_parent())
            now = time.monotonic()
            self._source_started_at[cid] = now
            self._watchdog_last_progress[cid] = now
            self._watchdog_last_frames[cid] = int(self.stats[cid].frames)
            self._source_recycling.discard(cid)
            print(
                "CAMERA_POSE_SOURCE_RECYCLE "
                f"cid={cid} attempt={attempt}/{self._watchdog_max_recycles} "
                f"phase=PLAYING sync={int(ok)}",
                flush=True,
            )
            return False

        self.GLib.timeout_add(self._watchdog_recycle_pause_ms, _restart_one)

    def _watch_sources(self) -> bool:
        if self._stopping:
            return False

        now = time.monotonic()
        for camera in self.cameras:
            cid = camera.camera_id
            started_at = self._source_started_at.get(cid)
            if started_at is None or now - started_at < self._watchdog_grace_s:
                continue
            if cid in self._source_recycling:
                continue

            frames = int(self.stats[cid].frames)
            previous = int(self._watchdog_last_frames.get(cid, frames))
            if frames > previous:
                self._watchdog_last_frames[cid] = frames
                self._watchdog_last_progress[cid] = now
                if self._source_recycle_count.get(cid, 0):
                    print(
                        "CAMERA_POSE_SOURCE_RECOVERED "
                        f"cid={cid} frames={frames} recycle_count={self._source_recycle_count[cid]}",
                        flush=True,
                    )
                self._source_recycle_count[cid] = 0
                continue

            stalled_s = now - self._watchdog_last_progress.get(cid, started_at)
            if stalled_s < self._watchdog_stall_s:
                continue

            used = int(self._source_recycle_count.get(cid, 0))
            if used < self._watchdog_max_recycles:
                self._source_recycle_count[cid] = used + 1
                self._watchdog_last_progress[cid] = now
                self._hard_recycle_source(cid, stalled_s)
                continue

            self._restart_requested = True
            self._restart_reason = (
                f"{cid} stalled {stalled_s:.1f}s after {used} hard recycle attempts"
            )
            print(
                "CAMERA_POSE_PROCESS_RESTART "
                f"reason={self._restart_reason!r} exit_code={RESTART_EXIT_CODE}",
                flush=True,
            )
            self.stop()
            return False

        return True

    def run(self) -> int:
        self._schedule_staggered_sources()
        self.GLib.timeout_add_seconds(1, self._watch_sources)
        result = super().run()
        if self._restart_requested:
            return RESTART_EXIT_CODE
        return result


def main() -> int:
    return CameraPersonTrackingPoseStickyV2().run()


if __name__ == "__main__":
    raise SystemExit(main())
