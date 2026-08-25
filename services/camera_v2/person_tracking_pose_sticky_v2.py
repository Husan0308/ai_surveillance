from __future__ import annotations

"""Verified pose-sticky runtime.

The historical CameraPersonTrackingFinal._stabilize_tracker_config() rewrites the
NvDCF YAML after tracker_profile.prepare_sparse_tracker_config().  That silently
restored probationAge=1 / earlyTerminationAge=2 and defeated the sparse ~1 Hz
pose design.  This subclass owns the *final* YAML passed to nvtracker and verifies
its effective values before the plugin is constructed.
"""

from pathlib import Path

from .person_tracking_pose_sticky import CameraPersonTrackingPoseSticky


_EFFECTIVE = {
    "minDetectorConfidence": "0.05",
    "minTrackerConfidence": "0.08",
    "probationAge": "0",
    "maxShadowTrackingAge": "50",
    "earlyTerminationAge": "6",
    "tentativeDetectorConfidence": "0.05",
    "minTrackingConfidenceDuringInactive": "0.05",
    "outputShadowTracks": "1",
}


def _rewrite_yaml_keys(path: Path, values: dict[str, str]) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    seen: dict[str, str] = {}
    output: list[str] = []

    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        replaced = False
        for key, value in values.items():
            if stripped.startswith(key + ":"):
                comment = ""
                if "#" in stripped:
                    comment = "  #" + stripped.split("#", 1)[1]
                output.append(f"{indent}{key}: {value}{comment}")
                seen[key] = value
                replaced = True
                break
        if not replaced:
            output.append(line)

    missing = sorted(set(values) - set(seen))
    if missing:
        raise RuntimeError(
            "pose NvDCF config is missing required keys after generation: "
            + ", ".join(missing)
        )

    path.write_text("\n".join(output) + "\n", encoding="utf-8")

    # Re-read the exact file nvtracker will consume.  Do not trust startup labels.
    effective: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        body = line.split("#", 1)[0].strip()
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        key = key.strip()
        if key in values:
            effective[key] = value.strip()

    wrong = {
        key: (values[key], effective.get(key))
        for key in values
        if effective.get(key) != values[key]
    }
    if wrong:
        raise RuntimeError(f"pose NvDCF effective config mismatch: {wrong}")
    return effective


class CameraPersonTrackingPoseStickyV2(CameraPersonTrackingPoseSticky):
    """Pose refresh + verified immediate-active NvDCF local tracking."""

    @staticmethod
    def _stabilize_tracker_config(path: Path) -> Path:
        # Intentionally do NOT call CameraPersonTrackingFinal's implementation:
        # that legacy method is the component that changes probation back to 1.
        path = Path(path)
        effective = _rewrite_yaml_keys(path, _EFFECTIVE)
        print(
            "CAMERA_POSE_NVDCF_EFFECTIVE "
            + " ".join(f"{key}={effective[key]}" for key in _EFFECTIVE),
            flush=True,
        )
        return path


def main() -> int:
    return CameraPersonTrackingPoseStickyV2().run()


if __name__ == "__main__":
    raise SystemExit(main())
