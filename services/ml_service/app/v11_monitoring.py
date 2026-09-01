from __future__ import annotations

import os
import threading
import time
from typing import Any

from services.camera_v11.step2_production_fp32 import _pct
from services.camera_v11.step3_tracking_v2 import V11Step3TrackingV2


class V11MonitoringTrackerRuntime(V11Step3TrackingV2):
    """Read-only publisher around the accepted V11 detector + local tracker path.

    The V11 detector keeps its existing substream/cadence. This subclass only
    copies the latest tracker snapshots into a bounded metadata slot for API/UI
    consumption; no extra inference or frame queue is introduced.
    """

    def __init__(self) -> None:
        super().__init__()
        self._monitor_lock = threading.Lock()
        self._monitor_seq = 0
        self._monitor_rows: dict[str, dict[str, Any]] = {
            camera.camera_id: {
                "frame_seq": 0,
                "timestamp_ns": 0,
                "tracks": [],
            }
            for camera in self.cameras
        }

    def _consume_tracking(self, cid: str, boxes: list[list[float]], captured_ns: int) -> None:
        # Keep the frozen Step3 bookkeeping/semantics exactly the same; the only
        # addition is copying the already-computed snapshots to a latest-only slot.
        update = self.tracker.update(cid, boxes, captured_ns)
        self.stage_values["tracker"].append(float(update.step_ms))
        ids = tuple(snapshot.track_id for snapshot in update.snapshots)
        if len(ids) != len(set(ids)):
            self.track_duplicate_errors += 1
        prefix = f"{cid}-T"
        self.track_prefix_errors += sum(1 for track_id in ids if not track_id.startswith(prefix))
        self.track_updates[cid] += 1
        self.track_created[cid] += int(update.created)
        self.track_recovered[cid] += int(update.recovered)
        self.track_removed[cid] += int(update.removed)
        self.latest_track_ids[cid] = ids

        tracks: list[dict[str, Any]] = []
        for snapshot in update.snapshots:
            if not snapshot.confirmed or snapshot.state == "removed":
                continue
            tracks.append(
                {
                    "track_id": str(snapshot.track_id),
                    "class_name": "person",
                    "confidence": float(snapshot.score),
                    "state": str(snapshot.state),
                    "predicted": bool(snapshot.predicted),
                    "bbox_norm": [float(v) for v in snapshot.bbox_norm],
                    "velocity_norm_s": [float(v) for v in snapshot.velocity_norm_s],
                    "since_detection_sec": float(snapshot.since_detection_sec),
                }
            )

        with self._monitor_lock:
            self._monitor_seq += 1
            self._monitor_rows[cid] = {
                "frame_seq": int(self._monitor_seq),
                "timestamp_ns": int(captured_ns),
                "tracks": tracks,
            }

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

    def monitoring_snapshot(self, camera_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        now_ns = time.monotonic_ns()
        metrics_by_id = {str(row.get("id")): row for row in camera_metrics}
        with self._monitor_lock:
            rows = {
                cid: {
                    "frame_seq": int(value["frame_seq"]),
                    "timestamp_ns": int(value["timestamp_ns"]),
                    "tracks": [dict(track) for track in value["tracks"]],
                }
                for cid, value in self._monitor_rows.items()
            }

        items: list[dict[str, Any]] = []
        for camera in self.cameras:
            cid = camera.camera_id
            metric = metrics_by_id.get(cid, {})
            width = max(1, int(metric.get("width") or 672))
            height = max(1, int(metric.get("height") or 384))
            row = rows[cid]
            age_sec = (
                max(0.0, (now_ns - row["timestamp_ns"]) / 1_000_000_000.0)
                if row["timestamp_ns"] > 0
                else 0.0
            )
            # Read-only bounded extrapolation makes the overlay move between the
            # sparse 2 Hz detector updates without creating a second tracker.
            predict_dt = min(0.45, age_sec)
            tracks: list[dict[str, Any]] = []
            for track in row["tracks"]:
                norm = self._predict_bbox_norm(track, predict_dt)
                # The V11 detector canvas is 672x384 with the real 16:9 content
                # occupying rows 3..380 (378 px). Remove that detector padding
                # before mapping to the native/main-stream frame.
                x1n = max(0.0, min(1.0, norm[0]))
                x2n = max(x1n, min(1.0, norm[2]))
                y1n = max(0.0, min(1.0, (norm[1] * 384.0 - 3.0) / 378.0))
                y2n = max(y1n, min(1.0, (norm[3] * 384.0 - 3.0) / 378.0))
                tracks.append(
                    {
                        "track_id": track["track_id"],
                        "class_name": "person",
                        "confidence": float(track["confidence"]),
                        "state": "predicted" if predict_dt > 0.075 or track.get("predicted") else track["state"],
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
                    "online": bool(metric.get("online", False)),
                    "fps": float(metric.get("fps") or 0.0),
                    "last_error": metric.get("last_error"),
                    "tracks": tracks,
                }
            )

        return {
            "type": "monitoring",
            "generated_ns": now_ns,
            "items": items,
        }


class V11MonitoringTrackerService:
    """Lifecycle wrapper so FastAPI can own the V11 tracker without blocking."""

    def __init__(self) -> None:
        self.enabled = os.getenv("ML_V11_TRACKING_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        self._thread: threading.Thread | None = None
        self._runtime: V11MonitoringTrackerRuntime | None = None
        self._lock = threading.Lock()
        self._last_error: str | None = None

    def start(self) -> None:
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._thread = threading.Thread(target=self._run, name="v11-monitoring-tracker", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        runtime: V11MonitoringTrackerRuntime | None = None
        try:
            runtime = V11MonitoringTrackerRuntime()
            with self._lock:
                self._runtime = runtime
                self._last_error = None
            runtime.run()
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            print(f"V11_MONITORING_TRACKER_ERROR {type(exc).__name__}: {exc}", flush=True)
        finally:
            if runtime is not None:
                try:
                    runtime.close()
                except Exception:
                    pass
            with self._lock:
                self._runtime = None

    def stop(self) -> None:
        with self._lock:
            runtime = self._runtime
        if runtime is not None:
            runtime.stop_requested = True
        if self._thread:
            self._thread.join(timeout=4.0)
            self._thread = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            runtime = self._runtime
            error = self._last_error
        return {
            "enabled": self.enabled,
            "ready": runtime is not None,
            "last_error": error,
        }

    def snapshot(self, camera_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            runtime = self._runtime
            error = self._last_error
        if runtime is None:
            items = []
            for metric in camera_metrics:
                items.append(
                    {
                        "camera_id": str(metric.get("id", "")),
                        "frame_seq": 0,
                        "timestamp_ns": 0,
                        "source_width": max(1, int(metric.get("width") or 1)),
                        "source_height": max(1, int(metric.get("height") or 1)),
                        "online": bool(metric.get("online", False)),
                        "fps": float(metric.get("fps") or 0.0),
                        "last_error": error or metric.get("last_error"),
                        "tracks": [],
                    }
                )
            return {"type": "monitoring", "generated_ns": time.monotonic_ns(), "items": items}
        return runtime.monitoring_snapshot(camera_metrics)
