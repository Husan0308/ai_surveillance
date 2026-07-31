import time
from datetime import datetime

from PySide6.QtCore import QObject, Signal, QTimer

from backend.core.system_monitor import SystemMonitor
from backend.core.logger import get_logger

log = get_logger("features.analytics")


class AnalyticsService(QObject):
    """
    Analytics data provider.

    Real-time:
        occupancy
        known / unknown
        detection count
        recognition count
        stay duration
        camera utilization
        GPU / CPU / RAM
        FPS
        peak hour
    """

    analytics_updated = Signal(dict)

    def __init__(
        self,
        config,
        db,
        db_writer=None,
        identity_manager=None,
        events_service=None,
    ):
        super().__init__()

        self.config = config
        self.db = db
        self.db_writer = db_writer

        self.identity_manager = identity_manager
        self.events_service = events_service

        self.monitor = SystemMonitor()

        # camera states
        self.camera_online = {}
        self.camera_fps = {}
        self.camera_first_seen = {}
        self.utilization_seconds = {}

        self.start_time = time.time()

        # real-time series
        self.max_series = 90

        self.occupancy_series = []
        self.gpu_series = []
        self.fps_series = []

        # cache
        self._peak_cache = None
        self._peak_ts = 0.0

        self._db_detection_total = 0
        self._db_recognition_total = 0
        self._db_totals_ts = 0.0

        # ✅ Global unikal identity tracking (cross-camera)
        self._known_person_ids = {}    # person_id → last_seen_time
        self._unknown_tracks = {}      # (camera_id, track_name) → last_seen_time
        self._identity_ttl = 60.0      # 60 sek ko'rinmasa → o'chirish

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        # ✅ EventsService dan identity event larini tinglash
        if events_service is not None:
            try:
                events_service.event_added.connect(self._on_identity_event)
            except Exception as e:
                log.error("analytics event connect error: %s", e)

        log.info("AnalyticsService started")

    # ---------------- external inputs ----------------
    def set_camera_online(self, camera_id: str, online: bool):
        self.camera_online[camera_id] = bool(online)

        if camera_id not in self.camera_first_seen:
            self.camera_first_seen[camera_id] = time.time()

    def set_camera_fps(self, camera_id: str, fps: float):
        self.camera_fps[camera_id] = float(fps)

    # ---------------- helpers ----------------
    def _get_peak_hour(self):
        """
        Peak hour: bugungi eng ko'p odam ko'rilgan soat.
        Har soat 60 minut. Faqat o'tgan soatlar hisobga olinadi.
        """
        now = time.time()

        if self._peak_cache is not None and now - self._peak_ts < 30.0:
            return self._peak_cache

        try:
            from datetime import datetime
            date_str = datetime.now().strftime("%Y-%m-%d")
            current_hour = datetime.now().hour

            # DB dan soatlik ma'lumot
            raw = self.db.get_peak_hour(date_str)

            if raw and len(raw) == 24:
                # Faqat o'tgan soatlar (hozirgi soatni hisobga olmaslik)
                peak_data = list(raw)
                # Hozirgi va kelgusi soatlarni 0 qilish
                for h in range(current_hour + 1, 24):
                    peak_data[h] = 0
                self._peak_cache = peak_data
            else:
                # DB dan ma'lumot yo'q → real-time dan hisoblash
                peak_data = [0] * 24
                # IdentityManager dan hozirgi occupancy ni olish
                if self.identity_manager is not None:
                    total = 0
                    for cam_id, state in self.identity_manager.states.items():
                        total += state.occupancy
                    peak_data[current_hour] = total
                self._peak_cache = peak_data

            self._peak_ts = now

        except Exception as e:
            log.error("get_peak_hour error: %s", e)
            if self._peak_cache is None:
                self._peak_cache = [0] * 24

        return self._peak_cache


    def _get_db_totals(self):
        now = time.time()

        if now - self._db_totals_ts > 10.0:
            try:
                det, rec = self.db.get_detection_recognition_today()
                self._db_detection_total = det
                self._db_recognition_total = rec
                self._db_totals_ts = now
            except Exception as e:
                log.error("get_db_totals error: %s", e)

        return self._db_detection_total, self._db_recognition_total

    def _avg_stay_sec(self):
        stays = []

        if self.identity_manager is not None:
            for state in self.identity_manager.states.values():
                if state.closed_stays:
                    stays.extend(state.closed_stays[-50:])

        if stays:
            return sum(stays) / float(len(stays))

        try:
            return self.db.get_avg_stay_today()
        except Exception:
            return 0.0

    def _camera_utilization(self):
        now = time.time()
        out = {}

        for camera_id, online in self.camera_online.items():
            if online:
                self.utilization_seconds[camera_id] = (
                    self.utilization_seconds.get(camera_id, 0) + 1
                )

            first_seen = self.camera_first_seen.get(camera_id, self.start_time)
            elapsed = max(1.0, now - first_seen)

            seconds = self.utilization_seconds.get(camera_id, 0)

            out[camera_id] = round((seconds / elapsed) * 100.0, 1)

        return out

    # ---------------- main tick ----------------

    # ---------------- identity tracking ----------------
    def _on_identity_event(self, event: dict):
        """EventsService dan identity event larini qabul qilish"""
        now = time.time()
        etype = event.get('type', '')
        if etype == 'person_recognized':
            pid = event.get('person_id')
            if pid is not None:
                self._known_person_ids[pid] = now
        elif etype == 'unknown_detected':
            cam = event.get('camera_id', '?')
            name = event.get('person_name', '?')
            self._unknown_tracks[(cam, name)] = now

    def _cleanup_identities(self):
        """60 sek ko'rinmagan identity larni o'chirish"""
        now = time.time()
        ttl = getattr(self, '_identity_ttl', 60.0)
        self._known_person_ids = {
            k: v for k, v in self._known_person_ids.items() if now - v < ttl
        }
        self._unknown_tracks = {
            k: v for k, v in self._unknown_tracks.items() if now - v < ttl
        }

    def _tick(self):
        occupancy = 0
        live_detections = 0
        live_recognitions = 0
        fps_values = []

        if self.identity_manager is not None:
            for camera_id, state in self.identity_manager.states.items():
                occupancy += state.occupancy
                live_detections += state.detections_total
                live_recognitions += state.recognitions_total

        # ✅ Eski identity larni tozalash
        self._cleanup_identities()

        # ✅ Known: unikal person_id lar (cross-camera, 1 odam = 1 known)
        known = len(self._known_person_ids)

        # ✅ Unknown: unikal track lar (har kamera alohida, 4 track = 4 unknown)
        unknown = len(self._unknown_tracks)

        for camera_id, fps in self.camera_fps.items():
            if fps > 0:
                fps_values.append(fps)

        avg_fps = sum(fps_values) / float(len(fps_values)) if fps_values else 0.0

        utilization = self._camera_utilization()

        stats = self.monitor.sample()

        db_det, db_rec = self._get_db_totals()

        detections_total = max(live_detections, db_det)
        recognitions_total = max(live_recognitions, db_rec)

        avg_stay = self._avg_stay_sec()

        peak_hour = self._get_peak_hour()

        # series
        self.occupancy_series.append(occupancy)
        self.gpu_series.append(stats["gpu"])
        self.fps_series.append(avg_fps)

        if len(self.occupancy_series) > self.max_series:
            self.occupancy_series.pop(0)

        if len(self.gpu_series) > self.max_series:
            self.gpu_series.pop(0)

        if len(self.fps_series) > self.max_series:
            self.fps_series.pop(0)

        cameras_online = sum(1 for v in self.camera_online.values() if v)
        cameras_total = len(self.camera_online)

        data = {
            "time": datetime.now().isoformat(),

            "occupancy": occupancy,
            "known": known,
            "unknown": unknown,

            "detections_total": detections_total,
            "recognitions_total": recognitions_total,

            "avg_stay_sec": round(avg_stay, 1),

            "peak_hour": peak_hour,

            "camera_utilization": utilization,
            "camera_fps": dict(self.camera_fps),

            "avg_fps": round(avg_fps, 1),

            "cpu": stats["cpu"],
            "ram": stats["ram"],
            "gpu": stats["gpu"],

            "cameras_online": cameras_online,
            "cameras_total": cameras_total,

            "occupancy_series": list(self.occupancy_series),
            "gpu_series": list(self.gpu_series),
            "fps_series": list(self.fps_series),
        }

        self.analytics_updated.emit(data)

    # ---------------- on-demand ----------------
    def get_current_data(self):
        """
        UI istalgan vaqtda so'rasa ishlatiladi.
        """

        return {
            "occupancy_series": list(self.occupancy_series),
            "gpu_series": list(self.gpu_series),
            "fps_series": list(self.fps_series),
            "peak_hour": self._get_peak_hour(),
            "camera_utilization": self._camera_utilization(),
            "camera_fps": dict(self.camera_fps),
            "camera_online": dict(self.camera_online),
        }

    # ---------------- shutdown ----------------
    def shutdown(self):
        self._timer.stop()
        log.info("AnalyticsService stopped")