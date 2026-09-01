from __future__ import annotations

import os
import threading
import time
from typing import Any

import cv2

from services.camera_v11.step3_tracker_v2 import V11PerCameraTrackerV2
from services.ml_service.app.detector_only_shm import CONTENT_H, INPUT_W, TRT86DetectorClient


class V11MonitoringTrackerService:
    """Latest-only detector/tracker consumer of the existing ML camera runtime.

    This service NEVER opens RTSP/NVDEC/DeepStream sources. The canonical
    DeepStreamRuntime remains the only camera owner. We sample its per-camera
    LatestFrameStore at a bounded detector cadence, run the one shared TRT86
    detector, then feed the existing V11 per-camera local tracker. The resulting
    metadata is published from a single latest-only slot per camera for API/UI.
    """

    def __init__(self, camera_runtime) -> None:
        self.camera_runtime = camera_runtime
        self.camera_ids = tuple(str(camera.camera_id) for camera in camera_runtime.settings.cameras)
        self.enabled = os.getenv("ML_V11_TRACKING_ENABLED", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        self.target_hz = max(0.2, min(5.0, float(os.getenv("ML_V11_TRACKING_HZ", "2.0"))))
        self.target_period = 1.0 / self.target_hz
        self.conf = min(1.0, max(0.01, float(os.getenv("ML_V11_TRACKING_CONF", "0.18"))))
        self.max_det = max(1, min(100, int(os.getenv("ML_V11_TRACKING_MAX_DET", "20"))))
        self.max_input_age_ms = max(
            50.0, float(os.getenv("ML_V11_TRACKING_MAX_INPUT_AGE_MS", "240"))
        )
        self.monitor_stale_sec = max(
            0.6, float(os.getenv("ML_V11_MONITOR_STALE_SEC", "1.50"))
        )

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_error: str | None = None
        self._ready = False
        self._detector: TRT86DetectorClient | None = None
        self._tracker: V11PerCameraTrackerV2 | None = None
        self._last_versions = {cid: 0 for cid in self.camera_ids}
        self._next_due = {cid: 0.0 for cid in self.camera_ids}
        self._sequence = 0
        self._processed = {cid: 0 for cid in self.camera_ids}
        self._stale_skips = {cid: 0 for cid in self.camera_ids}
        self._rows: dict[str, dict[str, Any]] = {
            cid: {"frame_seq": 0, "timestamp_ns": 0, "tracks": []}
            for cid in self.camera_ids
        }

    def start(self) -> None:
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="v11-monitoring-tracker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        detector: TRT86DetectorClient | None = None
        try:
            detector = TRT86DetectorClient()
            tracker = V11PerCameraTrackerV2(self.camera_ids)
            with self._lock:
                self._detector = detector
                self._tracker = tracker
                self._last_error = None
                self._ready = True
            print(
                "V11_MONITORING_TRACKER_READY "
                f"source=ml-latest-frame-store rtsp=0 nvdec=0 extra_camera_pipeline=0 "
                f"detector=shared-trt86 cadence={self.target_hz:.2f}Hz/cam tracker=v11-step3",
                flush=True,
            )

            cursor = 0
            while not self._stop.is_set():
                did_work = False
                for offset in range(len(self.camera_ids)):
                    idx = (cursor + offset) % len(self.camera_ids)
                    cid = self.camera_ids[idx]
                    now = time.monotonic()
                    if now < self._next_due[cid]:
                        continue
                    self._next_due[cid] = now + self.target_period
                    frame, version = self.camera_runtime.stores[cid].get()
                    if frame is None or version <= self._last_versions[cid]:
                        continue
                    self._last_versions[cid] = version
                    age_ms = max(0.0, (time.monotonic() - float(frame.captured_monotonic)) * 1000.0)
                    if age_ms > self.max_input_age_ms:
                        self._stale_skips[cid] += 1
                        continue

                    did_work = True
                    # Detector input is exactly the established 672x378 content;
                    # TRT86DetectorClient adds the existing 3px top/bottom padding.
                    detector_frame = cv2.resize(
                        frame.image,
                        (INPUT_W, CONTENT_H),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    result = detector.infer(detector_frame, self.conf, self.max_det)
                    captured_ns = int(float(frame.captured_monotonic) * 1_000_000_000.0)
                    update = tracker.update(cid, result.boxes, captured_ns)
                    tracks: list[dict[str, Any]] = []
                    for snapshot in update.snapshots:
                        if not snapshot.confirmed or snapshot.state == "removed":
                            continue
                        tracks.append(
                            {
                                "track_id": str(snapshot.track_id),
                                "confidence": float(snapshot.score),
                                "state": str(snapshot.state),
                                "predicted": bool(snapshot.predicted),
                                "bbox_norm": [float(v) for v in snapshot.bbox_norm],
                                "velocity_norm_s": [float(v) for v in snapshot.velocity_norm_s],
                            }
                        )
                    with self._lock:
                        self._sequence += 1
                        self._processed[cid] += 1
                        self._rows[cid] = {
                            "frame_seq": self._sequence,
                            "timestamp_ns": captured_ns,
                            "tracks": tracks,
                        }
                cursor = (cursor + 1) % max(1, len(self.camera_ids))
                if not did_work:
                    time.sleep(0.002)
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            print(f"V11_MONITORING_TRACKER_ERROR {type(exc).__name__}: {exc}", flush=True)
        finally:
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass
            with self._lock:
                self._detector = None
                self._tracker = None
                self._ready = False

    @staticmethod
    def _predict_bbox_norm(track: dict[str, Any], dt: float) -> list[float]:
        x1, y1, x2, y2 = (float(v) for v in track["bbox_norm"])
        vx, vy, vw, vh = (float(v) for v in track.get("velocity_norm_s", (0, 0, 0, 0)))
        cx = 0.5 * (x1 + x2) + vx * dt
        cy = 0.5 * (y1 + y2) + vy * dt
        width = max(0.002, (x2 - x1) + vw * dt)
        height = max(0.002, (y2 - y1) + vh * dt)
        px1 = max(0.0, min(1.0, cx - 0.5 * width))
        py1 = max(0.0, min(1.0, cy - 0.5 * height))
        px2 = max(px1, min(1.0, cx + 0.5 * width))
        py2 = max(py1, min(1.0, cy + 0.5 * height))
        return [px1, py1, px2, py2]

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "ready": self._ready,
                "last_error": self._last_error,
                "source": "ml_latest_frame_store",
                "extra_rtsp_connections": 0,
                "target_hz": self.target_hz,
                "processed": dict(self._processed),
                "stale_skips": dict(self._stale_skips),
            }

    def snapshot(self, camera_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        now_ns = time.monotonic_ns()
        metrics_by_id = {str(row.get("id")): row for row in camera_metrics}
        with self._lock:
            rows = {
                cid: {
                    "frame_seq": int(value["frame_seq"]),
                    "timestamp_ns": int(value["timestamp_ns"]),
                    "tracks": [dict(track) for track in value["tracks"]],
                }
                for cid, value in self._rows.items()
            }
            ready = self._ready
            error = self._last_error

        items: list[dict[str, Any]] = []
        for cid in self.camera_ids:
            metric = metrics_by_id.get(cid, {})
            width = max(1, int(metric.get("width") or 1))
            height = max(1, int(metric.get("height") or 1))
            online = bool(metric.get("online", False))
            row = rows[cid]
            age_sec = (
                max(0.0, (now_ns - row["timestamp_ns"]) / 1_000_000_000.0)
                if row["timestamp_ns"] > 0
                else float("inf")
            )
            stale = (not online) or (not ready) or age_sec > self.monitor_stale_sec
            predict_dt = min(0.45, age_sec) if not stale else 0.0
            tracks: list[dict[str, Any]] = []
            for track in (() if stale else row["tracks"]):
                norm = self._predict_bbox_norm(track, predict_dt)
                # V11 tracker normalization is relative to 672x384 including the
                # detector's 3px top/bottom letterbox. Remove that before mapping
                # to native camera coordinates used by the API/UI.
                x1n = max(0.0, min(1.0, norm[0]))
                x2n = max(x1n, min(1.0, norm[2]))
                y1n = max(0.0, min(1.0, (norm[1] * 384.0 - 3.0) / 378.0))
                y2n = max(y1n, min(1.0, (norm[3] * 384.0 - 3.0) / 378.0))
                tracks.append(
                    {
                        "track_id": track["track_id"],
                        "class_name": "person",
                        "confidence": float(track["confidence"]),
                        "state": "predicted"
                        if predict_dt > 0.075 or track.get("predicted")
                        else track["state"],
                        "bbox_xyxy": [
                            x1n * width,
                            y1n * height,
                            x2n * width,
                            y2n * height,
                        ],
                    }
                )
            items.append(
                {
                    "camera_id": cid,
                    "frame_seq": int(row["frame_seq"]),
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "source_width": width,
                    "source_height": height,
                    "online": online,
                    "fps": float(metric.get("fps") or 0.0),
                    "last_error": error or metric.get("last_error"),
                    "metadata_age_ms": None if age_sec == float("inf") else age_sec * 1000.0,
                    "stale": stale,
                    "tracks": tracks,
                }
            )
        return {"type": "monitoring", "generated_ns": now_ns, "items": items}
