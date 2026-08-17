from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import threading
import time

import numpy as np
import yaml

from .person_tracking_final import CameraPersonTrackingFinal
from .reid_production import ProductionReIdIdentityEngine
from .reid_quality import bbox_iou
from .reid_runtime import CropJob


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "reid.yaml"
DISPLAY_HOLD_BINDING_SEC = 1.15


def _load_reid_config() -> dict:
    path = Path(os.environ.get("CAMERA_V2_REID_CONFIG", str(DEFAULT_CONFIG)))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(raw.get("reid") or raw)


class CameraPersonTrackingReID(CameraPersonTrackingFinal):
    """Stable Camera V2 local tracking plus an isolated Global-ID side path.

    The existing YOLO/NvDCF/display pipeline remains authoritative. ReID samples
    the already-existing sparse detector frame mailbox, pairs each frame with the
    nearest NvDCF metadata snapshot, and sends crops to bounded CPU workers. Slow
    or failing ReID/Qwen therefore cannot stall RTSP, YOLO, NvDCF or display.
    """

    def __init__(self) -> None:
        self.identity: ProductionReIdIdentityEngine | None = None
        self._reid_lock = threading.RLock()
        self._track_history: dict[str, deque[tuple[float, list[dict]]]] = {}
        self._sample_versions: dict[str, int] = {}
        self._last_real_track_seen: dict[tuple[str, int], float] = {}
        self._sample_stop = threading.Event()
        self._sample_thread: threading.Thread | None = None
        self._reid_crop_submitted = 0
        self._reid_snapshot_misses = 0
        self._reid_crop_failures = 0
        super().__init__()

        cfg = _load_reid_config()
        camera_rooms = {
            camera.camera_id: str(getattr(camera, "room", "") or "")
            for camera in self.cameras
        }
        self.identity = ProductionReIdIdentityEngine(
            camera_rooms, cfg, root=ROOT
        )
        self._track_history = {
            camera.camera_id: deque(maxlen=36) for camera in self.cameras
        }
        self._sample_versions = {camera.camera_id: 0 for camera in self.cameras}
        print(
            "CAMERA_GLOBAL_ID ready architecture=async-tracklet-reid "
            "crop_bank=10 top3_diverse=1 qwen_async=1 topology=config/cameras.yaml "
            "independent_gallery_check=1 gallery_poison_guard=1 rollback=1 "
            "same_camera_fragment_reconnect=1 sticky_display_hold=1 "
            "false_merge_policy=conservative",
            flush=True,
        )

    def _tracker_probe(self, pad, info):
        buffer = info.get_buffer()
        rows: list[dict] = []
        now = time.monotonic()
        if buffer is not None:
            try:
                # Snapshot only real current NvDCF targets BEFORE downstream display
                # hold/smoothing adds any visual-only objects.
                rows = self.bridge.copy_tracks(buffer, max_rows=128)
                grouped: dict[int, list[dict]] = {}
                current_keys: set[tuple[str, int]] = set()
                for row in rows:
                    source_id = int(row.get("source_id", -1))
                    grouped.setdefault(source_id, []).append(row)
                    if 0 <= source_id < len(self.cameras):
                        camera_id = self.cameras[source_id].camera_id
                        key = (camera_id, int(row["object_id"]))
                        current_keys.add(key)
                        self._last_real_track_seen[key] = now
                with self._reid_lock:
                    for source_id, camera in enumerate(self.cameras):
                        current = grouped.get(source_id, [])
                        self._track_history[camera.camera_id].append(
                            (now, [dict(x) for x in current])
                        )
                if self.identity is not None:
                    for source_id, camera in enumerate(self.cameras):
                        self.identity.observe_tracks(
                            camera.camera_id,
                            str(getattr(camera, "room", "") or ""),
                            grouped.get(source_id, []),
                            now=now,
                        )
            except Exception as exc:
                current_keys = set()
                self._reid_crop_failures += 1
                print(
                    f"CAMERA_GLOBAL_ID track_snapshot warning={type(exc).__name__}:{exc}",
                    flush=True,
                )
        else:
            current_keys = set()

        # Preserve the exact stable local NvDCF styling/counting path first. The
        # smoother may append a short display-only hold object using the old local
        # ID while a person is briefly hidden behind a chair/desk.
        result = super()._tracker_probe(pad, info)

        # Replace local Unknown_Cx_y for real current tracks and for the short
        # display-hold interval only. Do not apply long LOST bindings blindly:
        # that would be unsafe if a tracker ID were ever recycled later.
        if buffer is not None and self.identity is not None:
            try:
                bindings = self.identity.bindings()
                source_by_camera = {
                    camera.camera_id: source_id
                    for source_id, camera in enumerate(self.cameras)
                }
                mappings = []
                for (camera_id, object_id), binding in bindings.items():
                    key = (camera_id, int(object_id))
                    last_real = self._last_real_track_seen.get(key, -1e9)
                    if (
                        key not in current_keys
                        and now - last_real > DISPLAY_HOLD_BINDING_SEC
                    ):
                        continue
                    source_id = source_by_camera.get(camera_id)
                    if source_id is None:
                        continue
                    mappings.append(
                        {
                            "source_id": source_id,
                            "object_id": int(object_id),
                            "global_id": int(binding["global_id"]),
                            "state": binding.get("state", "TENTATIVE"),
                        }
                    )
                if mappings:
                    self.bridge.apply_global_track_style(buffer, mappings)
            except Exception as exc:
                print(
                    f"CAMERA_GLOBAL_ID label warning={type(exc).__name__}:{exc}",
                    flush=True,
                )
        return result

    def _closest_track_snapshot(
        self, cid: str, captured_t: float
    ) -> list[dict] | None:
        with self._reid_lock:
            history = list(self._track_history.get(cid, ()))
        if not history:
            return None
        best_t, best_rows = min(history, key=lambda row: abs(row[0] - captured_t))
        if abs(best_t - captured_t) > 0.42:
            return None
        return best_rows

    @staticmethod
    def _crop_from_track(
        frame: np.ndarray,
        row: dict,
        frame_width: int,
        frame_height: int,
    ):
        h, w = frame.shape[:2]
        sx = w / float(max(1, frame_width))
        sy = h / float(max(1, frame_height))
        x1 = float(row["left"]) * sx
        y1 = float(row["top"]) * sy
        x2 = (float(row["left"]) + float(row["width"])) * sx
        y2 = (float(row["top"]) + float(row["height"])) * sy
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        # Small full-body context helps clothing ReID without pulling neighbours in.
        x1 -= bw * 0.035
        x2 += bw * 0.035
        y1 -= bh * 0.020
        y2 += bh * 0.015
        ix1 = max(0, int(np.floor(x1)))
        iy1 = max(0, int(np.floor(y1)))
        ix2 = min(w, int(np.ceil(x2)))
        iy2 = min(h, int(np.ceil(y2)))
        if ix2 - ix1 < 8 or iy2 - iy1 < 16:
            return None, None
        return (
            frame[iy1:iy2, ix1:ix2].copy(),
            (float(ix1), float(iy1), float(ix2), float(iy2)),
        )

    def _sample_loop(self) -> None:
        while not self._sample_stop.is_set():
            did_work = False
            for camera in self.cameras:
                cid = camera.camera_id
                with self.mailbox.cv:
                    mailbox_row = self.mailbox.rows.get(cid)
                if mailbox_row is None:
                    continue
                version, captured_t, frame = mailbox_row
                if int(version) <= self._sample_versions.get(cid, 0):
                    continue
                self._sample_versions[cid] = int(version)
                did_work = True
                tracks = self._closest_track_snapshot(cid, float(captured_t))
                if tracks is None:
                    self._reid_snapshot_misses += 1
                    continue
                if self.identity is None:
                    continue

                frame_h, frame_w = frame.shape[:2]
                scaled_boxes: list[
                    tuple[int, tuple[float, float, float, float]]
                ] = []
                for index, row in enumerate(tracks):
                    _crop, scaled = self._crop_from_track(
                        frame, row, self.frame_width, self.frame_height
                    )
                    if scaled is not None:
                        scaled_boxes.append((index, scaled))

                for index, row in enumerate(tracks):
                    crop, scaled_bbox = self._crop_from_track(
                        frame, row, self.frame_width, self.frame_height
                    )
                    if crop is None or scaled_bbox is None:
                        continue
                    max_overlap = 0.0
                    for other_index, other_box in scaled_boxes:
                        if other_index == index:
                            continue
                        max_overlap = max(
                            max_overlap, bbox_iou(scaled_bbox, other_box)
                        )

                    det_conf = float(row.get("confidence", -1.0))
                    if det_conf < 0.0:
                        det_conf = 0.35
                    tracker_conf = float(row.get("tracker_confidence", -1.0))
                    accepted = self.identity.submit_crop(
                        CropJob(
                            camera_id=cid,
                            local_id=int(row["object_id"]),
                            room_id=str(getattr(camera, "room", "") or ""),
                            crop=crop,
                            source_bbox=scaled_bbox,
                            source_width=frame_w,
                            source_height=frame_h,
                            detector_confidence=det_conf,
                            tracker_confidence=tracker_conf,
                            max_other_iou=max_overlap,
                            captured_at=float(captured_t),
                        )
                    )
                    if accepted:
                        self._reid_crop_submitted += 1
            self._sample_stop.wait(0.008 if did_work else 0.035)

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        if self.identity is not None:
            metrics = self.identity.metrics()
            ident = metrics.get("identity", {})
            embed = metrics.get("embedder", {})
            qwen = metrics.get("qwen", {})
            print(
                "CAMERA_GLOBAL_ID "
                f"globals={ident.get('globals',0)} "
                f"confirmed={ident.get('confirmed_globals',0)} "
                f"states={ident.get('states',{})} "
                f"rollbacks={ident.get('rollbacks',0)} "
                f"gallery_skips={ident.get('gallery_update_skips',0)} "
                f"crops={self._reid_crop_submitted} "
                f"embedded={metrics.get('embedded',0)} "
                f"quality_rejects={metrics.get('quality_rejects',0)} "
                f"duplicates={metrics.get('duplicate_rejects',0)} "
                f"snapshot_miss={self._reid_snapshot_misses} "
                f"backend={embed.get('backend','pending')} "
                f"embed_ms={float(embed.get('last_batch_ms',0.0)):.1f} "
                f"qwen={int(bool(qwen.get('enabled')))} "
                f"qwen_calls={qwen.get('calls',0)} "
                f"qwen_ms={float(qwen.get('last_latency_ms',0.0)):.0f} "
                f"error={metrics.get('last_error') or 'none'}",
                flush=True,
            )
        return keep

    def run(self) -> int:
        assert self.identity is not None
        self.identity.start()
        self._sample_stop.clear()
        self._sample_thread = threading.Thread(
            target=self._sample_loop,
            name="camera-v2-reid-sampler",
            daemon=False,
        )
        self._sample_thread.start()
        try:
            return super().run()
        finally:
            self._sample_stop.set()
            if self._sample_thread is not None:
                self._sample_thread.join(timeout=2.0)
            self.identity.stop()


def main() -> int:
    return CameraPersonTrackingReID().run()


if __name__ == "__main__":
    raise SystemExit(main())
