from __future__ import annotations

"""Verified pose-sticky runtime for DeepStream 7.1.

CameraPersonTrackingFinal resolves the generated NvDCF YAML through
``self._stabilize_tracker_config``.  This subclass therefore owns the final file
passed to nvtracker and verifies the parameters that are part of the current
DeepStream 7.1 NvMultiObjectTracker configuration contract.

Two details matter here:
* ``tentativeDetectorConfidence`` is a current DataAssociator parameter, but the
  max-perf stock config may omit it.  We insert it into the correct section.
* ``minTrackingConfidenceDuringInactive`` belongs to older NvDCF configs and is
  not part of the DeepStream 7.1 parameter table, so its absence must not abort
  startup.  Continuity is controlled with probation/shadow/current tracker
  confidence parameters instead.

The six RTSP sources are also started deliberately one-by-one.  Camera V2 used to
carry ``startup_stagger_sec`` in config but transition the whole pipeline to
PLAYING in one shot, causing six nvurisrcbin sessions to hit the NVR together.
The pose runtime locks every source at NULL first, lets the parent pipeline enter
PLAYING, then unlocks/synchronizes CAM-01..CAM-06 at the configured interval.
"""

import os
from pathlib import Path

from .person_tracking_pose_sticky import CameraPersonTrackingPoseSticky


# Values that MUST exist in the exact final YAML consumed by nvtracker.
_REQUIRED_EFFECTIVE: dict[str, str] = {
    "minDetectorConfidence": "0.05",
    "minTrackerConfidence": "0.08",
    "probationAge": "0",
    "maxShadowTrackingAge": "50",
    "earlyTerminationAge": "6",
    "tentativeDetectorConfidence": "0.05",
    "outputShadowTracks": "1",
}

# DeepStream 7.1 section ownership for parameters that a stock max-perf file may
# legitimately omit.  Add them rather than treating their absence as corruption.
_SECTION_FOR_MISSING: dict[str, str] = {
    "tentativeDetectorConfidence": "DataAssociator",
    "outputShadowTracks": "TargetManagement",
}

# Older NvDCF releases exposed this parameter.  If a local stock config still
# contains it we tune it, but DeepStream 7.1 does not require it to be present.
_LEGACY_OPTIONAL: dict[str, str] = {
    "minTrackingConfidenceDuringInactive": "0.05",
}


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

    # Required current DS7.1 parameters that are absent from a lean stock config
    # are inserted into their documented modules.
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

    # Tune legacy fields only when the installed config happens to expose them.
    legacy_seen = _rewrite_existing(lines, _LEGACY_OPTIONAL)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Re-read the exact file nvtracker will consume.  Startup text is not proof.
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
    """Pose refresh + verified immediate-active NvDCF local tracking."""

    @staticmethod
    def _stabilize_tracker_config(path: Path) -> Path:
        # Do not call CameraPersonTrackingFinal's legacy stabilizer here.  This
        # method is the final policy owner before GstNvTracker consumes the YAML.
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

        # Keep RTSP bins out of the parent state transition.  Downstream mux,
        # tracker, tiler and sink may enter PLAYING immediately; each live source
        # is then attached to that already-running graph one at a time.
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
                print(
                    "CAMERA_POSE_SOURCE_START "
                    f"cid={camera_id} index={ordinal} sync={int(ok)}",
                    flush=True,
                )
                return False

            self.GLib.timeout_add(delay_ms, _start_one)

    def run(self) -> int:
        self._schedule_staggered_sources()
        return super().run()


def main() -> int:
    return CameraPersonTrackingPoseStickyV2().run()


if __name__ == "__main__":
    raise SystemExit(main())
