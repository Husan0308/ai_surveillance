from __future__ import annotations

import os
from pathlib import Path

from . import person_tracking_trt86_pose_gate as base
from .pose_gate_v3 import PoseGateClient


def _patch_key(lines: list[str], key: str, value: str, *, required: bool = False) -> bool:
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
        raise RuntimeError(f"ML V2 NvDCF config missing required key: {key}")
    return False


class CameraPersonTrackingTRT86PoseGateV2(base.CameraPersonTrackingTRT86PoseGate):
    """Pascal-safe low-latency detector + conservative pose + NvDCF runtime.

    This keeps the proven 1280x720 mux / 1920x720 wall geometry. The detector is
    refreshed often enough to acquire new people, while NvDCF is configured as a
    lightweight local visual tracker. Pose is only a sparse false-positive gate;
    one failed pose crop never deletes a plausible person by itself.
    """

    @staticmethod
    def _stabilize_tracker_config(path: Path) -> Path:
        path = base.CameraPersonTrackingTRT86PoseGate._stabilize_tracker_config(path)
        lines = path.read_text(encoding="utf-8").splitlines()

        # Duplicate suppression: NVIDIA documents lower minIouDiff4NewTarget as
        # the direction for preventing a second target on the same physical body.
        _patch_key(lines, "minIouDiff4NewTarget", "0.35", required=True)
        _patch_key(lines, "minIou4TargetDuplicate", "0.80")
        _patch_key(lines, "targetDuplicateRunInterval", "1")

        # Recall/continuity. Accepted detector boxes may be low-confidence after
        # pose validation, so the local visual tracker must not immediately hide
        # them. 120 shadow frames is ~6 s at a healthy 20 FPS.
        _patch_key(lines, "minTrackerConfidence", "0.08", required=True)
        _patch_key(lines, "probationAge", "0", required=True)
        _patch_key(lines, "maxShadowTrackingAge", "120", required=True)
        _patch_key(lines, "earlyTerminationAge", "6", required=True)
        _patch_key(lines, "minTrackingConfidenceDuringInactive", "0.05")

        # GTX 1050 Ti throughput: use the cheaper ColorNames visual feature only.
        # HOG + ColorNames is more expensive and the previous 448x256 workaround
        # traded away too much spatial accuracy. Keep 512x288 and simplify the
        # feature itself instead.
        color = _patch_key(lines, "useColorNames", "1")
        hog = _patch_key(lines, "useHog", "0")
        feature = _patch_key(lines, "featureImgSizeLevel", "3")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        text = path.read_text(encoding="utf-8")
        required_rows = (
            "minIouDiff4NewTarget: 0.35",
            "minTrackerConfidence: 0.08",
            "probationAge: 0",
            "maxShadowTrackingAge: 120",
            "earlyTerminationAge: 6",
        )
        missing = [row for row in required_rows if row not in text]
        if missing:
            raise RuntimeError("ML V2 NvDCF verification failed: " + ", ".join(missing))

        print(
            "CAMERA_ML_V2_NVDCF "
            "minIouDiff4NewTarget=0.35 minTrackerConfidence=0.08 "
            "probationAge=0 maxShadowTrackingAge=120 earlyTerminationAge=6 "
            f"ColorNames={'1' if color else 'unsupported'} "
            f"HOG={'0' if hog else 'unsupported'} "
            f"featureImgSizeLevel={'3' if feature else 'unsupported'}",
            flush=True,
        )
        return path

    def __init__(self) -> None:
        # Parent module imported PoseGateClient by value. Replace that symbol before
        # parent construction so exactly one worker is created and it is V3.
        base.PoseGateClient = PoseGateClient
        super().__init__()

        # Preserve the clear 1280x720 -> 1920x720 presentation. Only the expensive
        # source-to-mux scaler changes from Lanczos to bilinear. The final tiler
        # remains Lanczos, so the visible 640x360 tiles keep the sharp presentation
        # that was proven by the camera-only baseline.
        self._set_if(self.mux, "interpolation-method", 1)
        self._set_if(self.mux, "buffer-pool-size", 12)
        self._set_if(self.tiler, "interpolation-method", 4)

        tracker = getattr(self, "tracker", None)
        subbatch = "unsupported"
        if tracker is not None and tracker.find_property("sub-batches") is not None:
            # Two independent NvDCF low-level contexts reduce the six-stream
            # critical path. NVIDIA documents the equivalent 3:3 syntax.
            tracker.set_property("sub-batches", "3:3")
            if tracker.find_property("ll-config-file") is not None:
                tracker.set_property(
                    "ll-config-file",
                    f"{self.tracker_config};{self.tracker_config}",
                )
            if tracker.find_property("sub-batch-err-recovery-trial-cnt") is not None:
                tracker.set_property("sub-batch-err-recovery-trial-cnt", 3)
            subbatch = "3:3"

        if tracker is not None and tracker.find_property("operate-on-class-ids") is not None:
            tracker.set_property("operate-on-class-ids", "0")

        print(
            "CAMERA_ML_V2_RUNTIME "
            f"tracker={self.tracker_width}x{self.tracker_height} "
            f"sub_batches={subbatch} pose_gate=v3-two-hit-reject "
            "mux_scale=bilinear tiler_scale=lanczos display=1280x720->1920x720",
            flush=True,
        )


def main() -> int:
    return CameraPersonTrackingTRT86PoseGateV2().run()


if __name__ == "__main__":
    raise SystemExit(main())
