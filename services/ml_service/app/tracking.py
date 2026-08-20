from __future__ import annotations

from dataclasses import asdict, dataclass
import threading
import time

from services.ml_service.app.detector import DetectionStore


@dataclass(frozen=True, slots=True)
class Track:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class TrackSnapshot:
    camera_id: str
    frame_id: int
    captured_monotonic: float
    updated_monotonic: float
    tracks: tuple[Track, ...]

    @property
    def detections(self) -> tuple[Track, ...]:
        # LatestJpegPublisher intentionally consumes a generic object stream
        # exposing .detections; tracks can therefore replace raw detections
        # without coupling the presentation layer to ByteTrack internals.
        return self.tracks


class TrackStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[str, TrackSnapshot] = {}

    def put(self, snapshot: TrackSnapshot) -> None:
        with self._lock:
            self._rows[snapshot.camera_id] = snapshot

    def get(self, camera_id: str) -> TrackSnapshot | None:
        with self._lock:
            return self._rows.get(camera_id)

    def payload(self, camera_id: str) -> dict | None:
        snapshot = self.get(camera_id)
        if snapshot is None:
            return None
        now = time.monotonic()
        return {
            "camera_id": snapshot.camera_id,
            "frame_id": snapshot.frame_id,
            "people": len(snapshot.tracks),
            "age_ms": max(0.0, (now - snapshot.updated_monotonic) * 1000.0),
            "tracks": [asdict(row) for row in snapshot.tracks],
        }


@dataclass
class TrackerMetrics:
    state: str = "stopped"
    backend: str = "bytetrack"
    updates: int = 0
    active_tracks: int = 0
    created_tracks: int = 0
    last_update_ms: float = 0.0
    last_error: str = ""


class PersonTracker:
    """CPU-only per-camera ByteTrack stage.

    This is intentionally local-camera tracking only. It does not perform
    appearance matching, ReID, face recognition or cross-camera identity merge.
    Each camera owns an independent ByteTrack state and exposes local T-IDs.
    """

    def __init__(
        self,
        config,
        detections: DetectionStore,
        camera_ids: list[str],
        frame_width: int,
        frame_height: int,
        detector_fps: float,
    ) -> None:
        self.config = config
        self.detections = detections
        self.camera_ids = list(camera_ids)
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.detector_fps = float(detector_fps)
        self.results = TrackStore()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._metrics = TrackerMetrics(state="disabled" if not config.enabled else "stopped")
        self._camera_updates = {camera_id: 0 for camera_id in self.camera_ids}
        self._camera_active = {camera_id: 0 for camera_id in self.camera_ids}
        self._camera_created = {camera_id: 0 for camera_id in self.camera_ids}

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="person-tracker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 3.0) -> None:
        if self._thread:
            self._thread.join(timeout)

    def metrics(self) -> dict:
        with self._lock:
            payload = asdict(self._metrics)
        payload.update(
            {
                "enabled": self.enabled,
                "scope": "per-camera",
                "reid": False,
                "face": False,
                "track_buffer_seconds": float(self.config.track_buffer_seconds),
            }
        )
        return payload

    def camera_metrics(self, camera_id: str) -> dict:
        snapshot = self.results.get(camera_id)
        with self._lock:
            updates = int(self._camera_updates.get(camera_id, 0))
            active = int(self._camera_active.get(camera_id, 0))
            created = int(self._camera_created.get(camera_id, 0))
            state = self._metrics.state
            error = self._metrics.last_error
        return {
            "state": state,
            "backend": "bytetrack",
            "updates": updates,
            "active_tracks": active,
            "created_tracks": created,
            "frame_id": snapshot.frame_id if snapshot else 0,
            "last_error": error if state == "error" else "",
        }

    def snapshot_payload(self, camera_id: str) -> dict:
        return {"tracker": self.metrics(), "result": self.results.payload(camera_id)}

    def _set_error(self, exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._metrics.state = "error"
            self._metrics.last_error = message
        print(f"[TRACK] ERROR {message}", flush=True)

    def _run(self) -> None:
        try:
            import torch
            from ultralytics.engine.results import Boxes
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.utils import IterableSimpleNamespace

            buffer_frames = max(
                2,
                int(round(float(self.config.track_buffer_seconds) * max(0.1, self.detector_fps))),
            )
            args = {
                "tracker_type": "bytetrack",
                "track_high_thresh": float(self.config.track_high_thresh),
                "track_low_thresh": float(self.config.track_low_thresh),
                "new_track_thresh": float(self.config.new_track_thresh),
                "track_buffer": buffer_frames,
                "match_thresh": float(self.config.match_thresh),
                "fuse_score": bool(self.config.fuse_score),
            }

            # Construct every tracker before the first update. BYTETracker's
            # internal ID allocator is process-global; constructing lazily after
            # tracking has started would reset that allocator.
            trackers = {
                camera_id: BYTETracker(IterableSimpleNamespace(**dict(args)))
                for camera_id in self.camera_ids
            }
            local_maps: dict[str, dict[int, int]] = {camera_id: {} for camera_id in self.camera_ids}
            next_local = {camera_id: 1 for camera_id in self.camera_ids}
            last_frame_id = {camera_id: 0 for camera_id in self.camera_ids}

            with self._lock:
                self._metrics.state = "ready"
                self._metrics.last_error = ""
            print(
                f"[TRACK] ready backend=bytetrack cameras={len(self.camera_ids)} "
                f"buffer_frames={buffer_frames} reid=off face=off",
                flush=True,
            )

            while not self._stop.is_set():
                progressed = False
                for camera_id in self.camera_ids:
                    snapshot = self.detections.get(camera_id)
                    if snapshot is None or snapshot.frame_id <= last_frame_id[camera_id]:
                        continue
                    progressed = True
                    last_frame_id[camera_id] = int(snapshot.frame_id)

                    rows = [
                        [*detection.xyxy, float(detection.confidence), 0.0]
                        for detection in snapshot.detections
                    ]
                    if rows:
                        data = torch.tensor(rows, dtype=torch.float32)
                    else:
                        data = torch.empty((0, 6), dtype=torch.float32)
                    boxes = Boxes(data, (self.frame_height, self.frame_width))

                    started = time.perf_counter()
                    tracked = trackers[camera_id].update(boxes)
                    update_ms = (time.perf_counter() - started) * 1000.0

                    output: list[Track] = []
                    for row in tracked.tolist() if len(tracked) else ():
                        x1, y1, x2, y2, raw_id, score = row[:6]
                        raw_id_int = int(raw_id)
                        mapping = local_maps[camera_id]
                        local_id = mapping.get(raw_id_int)
                        if local_id is None:
                            local_id = next_local[camera_id]
                            next_local[camera_id] += 1
                            mapping[raw_id_int] = local_id
                            with self._lock:
                                self._camera_created[camera_id] += 1
                                self._metrics.created_tracks += 1
                        output.append(
                            Track(
                                track_id=local_id,
                                xyxy=(float(x1), float(y1), float(x2), float(y2)),
                                confidence=float(score),
                            )
                        )

                    now = time.monotonic()
                    self.results.put(
                        TrackSnapshot(
                            camera_id=camera_id,
                            frame_id=int(snapshot.frame_id),
                            captured_monotonic=float(snapshot.captured_monotonic),
                            updated_monotonic=now,
                            tracks=tuple(output),
                        )
                    )
                    with self._lock:
                        self._camera_updates[camera_id] += 1
                        self._camera_active[camera_id] = len(output)
                        self._metrics.updates += 1
                        self._metrics.active_tracks = sum(self._camera_active.values())
                        self._metrics.last_update_ms = update_ms

                if not progressed:
                    self._stop.wait(0.005)

        except BaseException as exc:
            self._set_error(exc)
        finally:
            if self._stop.is_set():
                with self._lock:
                    if self._metrics.state != "error":
                        self._metrics.state = "stopped"
