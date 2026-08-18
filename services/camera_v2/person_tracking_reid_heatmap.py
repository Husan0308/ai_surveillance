from __future__ import annotations

import os
from pathlib import Path
import threading
import time

# Primary detector stays on the proven GPU path. Pose is a tiny, low-rate CPU
# sidecar so it cannot steal CUDA time from YOLO26m/NvDCF display.
os.environ.setdefault("CAMERA_V2_DETECT_WIDTH", "736")
os.environ.setdefault("CAMERA_V2_DETECT_HEIGHT", "416")
os.environ.setdefault("CAMERA_V2_DETECT_CONF", "0.04")
os.environ.setdefault("CAMERA_V2_MAX_DET", "40")
os.environ.setdefault("CAMERA_V2_POSE_DEVICE", "cpu")
os.environ.setdefault("CAMERA_V2_POSE_MODEL", "yolo26n-pose.pt")
os.environ.setdefault("CAMERA_V2_POSE_TARGET_HZ", "0.80")

from .detector_latency import PreparedDetection
from .person_tracking_heatmap import CameraPersonTrackingHeatmap
from .pose_ankle import POSE_MIN_VISIBLE, PoseAnkleSidecar
from . import person_tracking_reid as _reid_module

# The native display smoother bridges up to ~1.8 s of sparse-detector misses.
# Keep Global-ID styling/counters alive for the same bounded visual interval.
_reid_module.DISPLAY_HOLD_BINDING_SEC = 1.90
CameraPersonTrackingReID = _reid_module.CameraPersonTrackingReID
DISPLAY_HOLD_BINDING_SEC = _reid_module.DISPLAY_HOLD_BINDING_SEC


class CameraPersonTrackingReIDHeatmap(
    CameraPersonTrackingReID,
    CameraPersonTrackingHeatmap,
):
    """Live tracking + ReID + ankle heatmap + conservative pose recovery.

    YOLO26m remains the primary detector and NvDCF remains the temporal tracker.
    YOLO26n-pose reuses already-decoded detector frames at 0.8 Hz/camera on CPU.
    It has two bounded jobs only:
      1. match ankle keypoints to an existing NvDCF track for heatmap deposition;
      2. inject a pose box only when no current tracker overlaps that pose person.
    """

    def __init__(self) -> None:
        self.pose_sidecar: PoseAnkleSidecar | None = None
        self.pose_ankle_matches = 0
        self.pose_ankle_misses = 0
        self.pose_recoveries = 0
        self.pose_recovery_skips = 0
        self._pose_lock = threading.RLock()
        super().__init__()
        self.pose_sidecar = PoseAnkleSidecar(
            mailbox=self.mailbox,
            camera_ids=[camera.camera_id for camera in self.cameras],
            on_result=self._consume_pose_result,
        )

    @staticmethod
    def _stabilize_tracker_config(path: Path) -> Path:
        """Tune NvDCF for difficult poses while keeping downstream IDs stable."""
        stabilized = CameraPersonTrackingReID._stabilize_tracker_config(path)
        replacements = {
            "minDetectorConfidence": "0.04",
            "maxShadowTrackingAge": "50",
            "earlyTerminationAge": "3",
            "minTrackingConfidenceDuringInactive": "0.15",
            "outputShadowTracks": "0",
        }
        lines = stabilized.read_text(encoding="utf-8").splitlines()
        found: set[str] = set()
        output: list[str] = []
        for line in lines:
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            replaced = False
            for key, value in replacements.items():
                if stripped.startswith(key + ":"):
                    comment = ""
                    if "#" in stripped:
                        comment = "  #" + stripped.split("#", 1)[1]
                    output.append(f"{indent}{key}: {value}{comment}")
                    found.add(key)
                    replaced = True
                    break
            if not replaced:
                output.append(line)

        missing = sorted(set(replacements) - found)
        if missing:
            raise RuntimeError(
                "Generated NvDCF config missing persistence keys: "
                + ", ".join(missing)
            )
        stabilized.write_text("\n".join(output) + "\n", encoding="utf-8")
        return stabilized

    @staticmethod
    def _track_box(row: dict) -> tuple[float, float, float, float]:
        left = float(row.get("left", 0.0))
        top = float(row.get("top", 0.0))
        width = max(0.0, float(row.get("width", 0.0)))
        height = max(0.0, float(row.get("height", 0.0)))
        return (left, top, left + width, top + height)

    def _publish_pose_recovery(
        self,
        cid: str,
        captured_t: float,
        boxes: list[tuple[float, float, float, float, float]],
    ) -> bool:
        if not boxes:
            return False
        prepared = [
            PreparedDetection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(conf),
                vx=0.0,
                vy=0.0,
            )
            for x1, y1, x2, y2, conf in boxes
        ]
        # Never overwrite a fresh primary YOLO observation that has not reached
        # nvtracker yet. Pose recovery waits for the next sparse opportunity.
        with self.pending_lock:
            current = self.pending.get(cid)
            if current is not None and int(current[0]) > int(self.injected_seq.get(cid, 0)):
                self.pose_recovery_skips += len(prepared)
                return False
            self.pending_seq += 1
            self.pending[cid] = (
                self.pending_seq,
                float(captured_t),
                prepared,
            )
        self.pose_recoveries += len(prepared)
        return True

    def _consume_pose_result(self, camera_result: dict) -> None:
        cid = str(camera_result.get("camera") or "")
        source_id = self.camera_index.get(cid)
        if source_id is None:
            return
        captured_t = float(camera_result.get("captured") or 0.0)
        frame_w = max(1, int(camera_result.get("frame_width") or 1))
        frame_h = max(1, int(camera_result.get("frame_height") or 1))
        rows = list(camera_result.get("rows") or [])
        tracks = self._closest_track_snapshot(cid, captured_t) or []

        sx = float(self.frame_width) / float(frame_w)
        sy = float(self.frame_height) / float(frame_h)
        current_track_boxes = [self._track_box(row) for row in tracks]
        recovery: list[tuple[float, float, float, float, float]] = []

        for pose in rows:
            raw_box = pose.get("box") or []
            if len(raw_box) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in raw_box]
            box = (
                max(0.0, x1 * sx),
                max(0.0, y1 * sy),
                min(float(self.frame_width - 1), x2 * sx),
                min(float(self.frame_height - 1), y2 * sy),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            best_index = -1
            best_iou = 0.0
            for index, track_box in enumerate(current_track_boxes):
                iou = self._det_iou(box, track_box)
                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            ankle = pose.get("ankle")
            if best_index >= 0 and best_iou >= 0.12 and ankle is not None and len(ankle) >= 3:
                track = tracks[best_index]
                nx = max(0.0, min(1.0, float(ankle[0]) / float(frame_w)))
                ny = max(0.0, min(1.0, float(ankle[1]) / float(frame_h)))
                quality = max(0.0, min(1.0, float(ankle[2])))
                self.deposit_pose_ankle(
                    source_id=int(source_id),
                    object_id=int(track["object_id"]),
                    captured_at=captured_t,
                    nx=nx,
                    ny=ny,
                    confidence=quality,
                )
                self.pose_ankle_matches += 1
                continue

            if ankle is not None:
                self.pose_ankle_misses += 1

            # Recovery is intentionally stricter than heatmap matching. Require a
            # genuine pose skeleton and no meaningful overlap with an existing
            # NvDCF person. This recovers reclining/foreshortened people without
            # creating duplicate boxes for already tracked workers.
            visible = int(pose.get("visible_keypoints") or 0)
            box_conf = float(pose.get("box_conf") or 0.0)
            area_ratio = (
                max(0.0, x2 - x1) * max(0.0, y2 - y1)
                / float(max(1, frame_w * frame_h))
            )
            if (
                best_iou < 0.12
                and visible >= POSE_MIN_VISIBLE
                and box_conf >= 0.12
                and area_ratio >= 0.0015
            ):
                duplicate = any(self._det_iou(box, r[:4]) >= 0.45 for r in recovery)
                if not duplicate:
                    recovery.append(
                        (
                            box[0],
                            box[1],
                            box[2],
                            box[3],
                            max(0.06, min(0.80, box_conf)),
                        )
                    )

        if recovery:
            self._publish_pose_recovery(cid, captured_t, recovery)

    def live_people_counts(self) -> dict[str, int]:
        """Return live unique Total/Known/Unknown counts used by the Qt cards."""
        now = time.monotonic()
        bindings = self.identity.bindings() if self.identity is not None else {}
        known_globals: set[int] = set()
        unknown_tracks: set[tuple[str, int]] = set()
        max_age = max(1.90, float(DISPLAY_HOLD_BINDING_SEC))

        with self._reid_lock:
            last_seen = dict(self._last_real_track_seen)

        for key, seen_at in last_seen.items():
            if now - float(seen_at) > max_age:
                continue
            binding = bindings.get(key)
            global_id = int((binding or {}).get("global_id") or 0)
            if global_id > 0:
                known_globals.add(global_id)
            else:
                unknown_tracks.add((str(key[0]), int(key[1])))

        known = len(known_globals)
        unknown = len(unknown_tracks)
        return {
            "total": known + unknown,
            "known": known,
            "unknown": unknown,
        }

    def active_people_count(self) -> int:
        return int(self.live_people_counts()["total"])

    def _tracker_probe(self, pad, info):
        result = super()._tracker_probe(pad, info)
        try:
            counts = self.live_people_counts()
            with self.det_lock:
                self.tracked_now = int(counts["total"])
        except Exception:
            pass
        return result

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        pose = self.pose_sidecar.metrics() if self.pose_sidecar is not None else {}
        print(
            "CAMERA_POSE "
            f"ready={int(bool(pose.get('ready')))} "
            f"device={pose.get('device','cpu')} model={pose.get('model','')} "
            f"calls={pose.get('calls',0)} batch={float(pose.get('batch_ms',0.0)):.1f}ms "
            f"ankle_matches={self.pose_ankle_matches} ankle_unmatched={self.pose_ankle_misses} "
            f"recoveries={self.pose_recoveries} recovery_skips={self.pose_recovery_skips} "
            f"error={pose.get('error') or 'none'}",
            flush=True,
        )
        return keep

    def run(self) -> int:
        assert self.pose_sidecar is not None
        self.pose_sidecar.start()
        try:
            return super().run()
        finally:
            self.pose_sidecar.stop()


def main() -> int:
    return CameraPersonTrackingReIDHeatmap().run()


if __name__ == "__main__":
    raise SystemExit(main())
