import time
from datetime import datetime

from PySide6.QtCore import QObject, Signal, QTimer

from backend.features.heatmap import HeatmapEngine
from backend.core.logger import get_logger

log = get_logger("features.identity")


class CameraIdentityState:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id

        # track_id -> info
        self.active_tracks = {}

        self.occupancy = 0
        self.known = 0
        self.unknown = 0

        self.detections_total = 0
        self.recognitions_total = 0

        self.closed_stays = []

        # analytics interval counters
        self.int_occupancy_sum = 0
        self.int_known = 0
        self.int_unknown = 0
        self.int_detections = 0
        self.int_recognitions = 0
        self.int_samples = 0

    @property
    def active_visits(self):
        return sum(1 for t in self.active_tracks.values() if t.get("visit_open"))

    @property
    def avg_stay_sec(self):
        if not self.closed_stays:
            return 0.0
        return sum(self.closed_stays) / float(len(self.closed_stays))


class IdentityManager(QObject):
    """
    Identity Manager.

    Responsibilities:
    - bind tracks to known persons
    - count known / unknown
    - manage visits
    - calculate stay duration
    - accumulate analytics
    - manage per-camera heatmap engines
    """

    metrics_updated = Signal(str, dict)       # camera_id, metrics
    identity_updated = Signal(str, object)
    persons_online = Signal(str, list)  # camera_id, [person_dicts]    # camera_id, AIResult

    def __init__(self, config, db, db_writer=None, event_bus=None):
        super().__init__()

        self.config = config
        self.db = db
        self.db_writer = db_writer
        self.event_bus = event_bus

        self.states = {}
        self.heatmaps = {}

        self._analytics_tick = 0

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._sample_1s)
        self._timer.start()

        log.info("IdentityManager started")

    # ---------------- public ----------------
    def get_state(self, camera_id: str) -> CameraIdentityState:
        if camera_id not in self.states:
            self.states[camera_id] = CameraIdentityState(camera_id)
        return self.states[camera_id]

    def get_heatmap(self, camera_id: str) -> HeatmapEngine:
        if camera_id not in self.heatmaps:
            self.heatmaps[camera_id] = HeatmapEngine(self.config, camera_id)
        return self.heatmaps[camera_id]

    def set_heatmap_on(self, camera_id: str, on: bool):
        self.get_heatmap(camera_id).set_on(on)

    def set_heatmap_mode(self, camera_id: str, mode: str):
        self.get_heatmap(camera_id).set_mode(mode)

    def reset_heatmap(self, camera_id: str, mode: str = None):
        self.get_heatmap(camera_id).reset(mode)

    def get_heatmap_image(self, camera_id: str, mode: str = None):
        return self.get_heatmap(camera_id).get_image(mode)

    # ---------------- main input ----------------
    def process_result(self, camera_id: str, result):
        """
        AIWorker.result_ready signaliga ulanadi.
        """
        state = self.get_state(camera_id)
        heatmap = self.get_heatmap(camera_id)

        frame = getattr(result, "frame", None)

        if frame is not None:
            frame_h, frame_w = frame.shape[:2]
        else:
            frame_w, frame_h = 640, 360

        persons = getattr(result, "persons", [])

        # heatmap update — even if visualization OFF
        heatmap.update(persons, frame_w, frame_h, online=True)

        now = time.time()
        if not hasattr(self, "_closed_by_person"):
            self._closed_by_person = {}   # camera_id -> person_id -> [duration,...] (DB siz, thread-safe)
        current = {}

        for p in persons:
            current[p.track_id] = p

            if p.track_id not in state.active_tracks:
                # new track
                state.detections_total += 1
                state.int_detections += 1

                visit_open = False

                if p.known and p.person_id is not None:
                    state.recognitions_total += 1
                    state.int_recognitions += 1

                    if self.db_writer is not None:
                        self.db_writer.submit(
                            "open_visit",
                            person_id=p.person_id,
                            camera_id=camera_id,
                            track_id=str(p.track_id),
                        )

                    visit_open = True

                state.active_tracks[p.track_id] = {
                    "person_id": p.person_id,
                    "known": bool(p.known),
                    "name": p.name,
                    "first_seen": now,
                    "last_seen": now,
                    "visit_open": visit_open,
                }

            else:
                info = state.active_tracks[p.track_id]
                info["last_seen"] = now

                # track became recognized later
                if p.known and p.person_id is not None and not info["known"]:
                    info["known"] = True
                    info["person_id"] = p.person_id
                    info["name"] = p.name

                    state.recognitions_total += 1
                    state.int_recognitions += 1

                    if not info["visit_open"] and self.db_writer is not None:
                        info["visit_open"] = True

                        self.db_writer.submit(
                            "open_visit",
                            person_id=p.person_id,
                            camera_id=camera_id,
                            track_id=str(p.track_id),
                        )

        # close disappeared tracks
        missing_ids = [tid for tid in state.active_tracks if tid not in current]

        for tid in missing_ids:
            info = state.active_tracks.pop(tid)

            if info.get("visit_open") and self.db_writer is not None:
                self.db_writer.submit(
                    "close_visit_by_track",
                    camera_id=camera_id,
                    track_id=str(tid),
                )

            duration = now - info.get("first_seen", now)

            if duration > 0:
                state.closed_stays.append(duration)
                if info.get("known") and info.get("person_id") is not None:
                    _cb = self._closed_by_person.setdefault(camera_id, {}).setdefault(info["person_id"], [])
                    _cb.append(duration)
                    if len(_cb) > 50:
                        self._closed_by_person[camera_id][info["person_id"]] = _cb[-50:]

                if len(state.closed_stays) > 200:
                    state.closed_stays.pop(0)

        # current metrics
        state.occupancy = len(current)
        state.known = sum(1 for p in current.values() if p.known)
        state.unknown = state.occupancy - state.known

        metrics = {
            "camera_id": camera_id,
            "occupancy": state.occupancy,
            "known": state.known,
            "unknown": state.unknown,
            "detections_total": state.detections_total,
            "recognitions_total": state.recognitions_total,
            "active_visits": state.active_visits,
            "avg_stay_sec": round(state.avg_stay_sec, 1),
            "infer_ms": round(getattr(result, "infer_ms", 0.0), 1),
            "heatmap_mode": heatmap.mode,
            "heatmap_on": heatmap.on,
        }

        # Online persons: active_tracks asosida BARQAROR (5s oyna -> miltillash YOQ)
        online_persons = []
        for tid, info in list(state.active_tracks.items()):
            if now - info.get("last_seen", 0) > 5.0:
                continue
            pid = info.get("person_id")
            if pid is None:
                continue
            stay_sec = max(0.0, now - info.get("first_seen", now))
            closed_sum = sum(self._closed_by_person.get(camera_id, {}).get(pid, []))
            total_stay = closed_sum + stay_sec
            online_persons.append({
                "person_id": pid,
                "name": info.get("name", "Unknown"),
                "known": bool(info.get("known")),
                "track_id": tid,
                "camera_id": camera_id,
                "stay_sec": round(stay_sec, 1),
                "total_stay": round(total_stay, 1),
            })
        self.persons_online.emit(camera_id, online_persons)
        self.metrics_updated.emit(camera_id, metrics)
        self.identity_updated.emit(camera_id, result)

    # ---------------- analytics sampling ----------------
    def _sample_1s(self):
        for state in self.states.values():
            state.int_occupancy_sum += state.occupancy
            state.int_known += state.known
            state.int_unknown += state.unknown
            state.int_samples += 1

        self._analytics_tick += 1

        # flush every 5 seconds
        if self._analytics_tick % 5 == 0:
            self._flush_analytics()

    def _flush_analytics(self):
        if self.db_writer is None:
            return

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        hour = now.hour

        for camera_id, state in self.states.items():
            if state.int_samples <= 0:
                continue

            self.db_writer.submit(
                "upsert_analytics_hourly",
                data={
                    "date": date_str,
                    "hour": hour,
                    "camera_id": camera_id,
                    "occupancy_sum": state.int_occupancy_sum,
                    "known_count": state.int_known,
                    "unknown_count": state.int_unknown,
                    "detection_count": state.int_detections,
                    "recognition_count": state.int_recognitions,
                },
            )

            state.int_occupancy_sum = 0
            state.int_known = 0
            state.int_unknown = 0
            state.int_detections = 0
            state.int_recognitions = 0
            state.int_samples = 0

    # ---------------- shutdown ----------------
    def shutdown(self):
        self._timer.stop()

        # close all active visits
        if self.db_writer is not None:
            for camera_id, state in self.states.items():
                for track_id, info in list(state.active_tracks.items()):
                    if info.get("visit_open"):
                        self.db_writer.submit(
                            "close_visit_by_track",
                            camera_id=camera_id,
                            track_id=str(track_id),
                        )

        log.info("IdentityManager stopped")