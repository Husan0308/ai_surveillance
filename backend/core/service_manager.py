import threading
from PySide6.QtCore import QObject, Signal

from backend.core.config import ConfigService
from backend.core.logger import setup_logging, get_logger
from backend.core.event_bus import EventBus
from backend.core.log_service import LogService
from backend.core.performance_monitor import PerformanceMonitor

from backend.db.database import Database
from backend.db.db_writer import DBWriter

from backend.cameras.camera_manager import CameraManager

from backend.ai.detector import Detector
from backend.ai.tracker import ByteTracker
from backend.ai.pose_engine import PoseEngine
from backend.ai.face_engine import FaceEngine
from backend.ai.reid_engine import ReIDEngine
from backend.ai.ai_worker import AIWorker
from backend.features.unknown_registry import UnknownRegistry
from backend.features.global_identity_cache import GlobalIdentityCache
from backend.features.room_manager import RoomManager

from backend.features.identity_manager import IdentityManager
from backend.features.enrollment import EnrollmentService
from backend.features.person_service import PersonService
from backend.features.unknown_service import UnknownService
from backend.features.events_service import EventsService
from backend.features.alerts_service import AlertsService
from backend.features.sanpshot_service import SnapshotService
from backend.features.analytics_service import AnalyticsService
from backend.features.settings_service import SettingsService

from backend.storage.recording_service import RecordingService
from backend.storage.cleanup_service import CleanupService
from backend.storage.export_service import ExportService
from backend.core.global_registry import GlobalIdentityRegistry
from backend.core.person_pool import GlobalPersonPool

log = get_logger("core.service_manager")


class ServiceManager(QObject):
    """
    Application service manager.

    Barcha backend servislarni yaratadi, bog'laydi va to'xtatadi.
    """

    ready = Signal()
    shutdown_finished = Signal()

    def __init__(self):
        super().__init__()
        

        # ---------------- core ----------------
        self.config = ConfigService()
        setup_logging(self.config)

        self.log_service = LogService(self.config)
        self.event_bus = EventBus()

        # ---------------- database ----------------
        self.db = Database(self.config)

        self.db_writer = DBWriter(
            self.db,
            maxsize=int(self.config.get("performance.db_queue_size", 10000)),
        )
        self.db_writer.start()

        # ✅ MUHIM: AI ENGINE'LAR BIRINCHI YARATILADI
        self.detector = Detector(self.config)
        self.pose_engine = PoseEngine(self.config)
        self.face_engine = FaceEngine(self.config, db=self.db)
        self.reid_engine = ReIDEngine(self.config)

        # 🌐 Cross-camera shared ReID gallery (bitta instance → hamma kamera ulashadi)
        try:
            import importlib.util as _iu, os as _os
            _gp = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "ai", "reid_gallery.py")
            _spec = _iu.spec_from_file_location("_reid_gallery_mod", _gp)
            _gm = _iu.module_from_spec(_spec); _spec.loader.exec_module(_gm)
            self.reid_gallery = _gm.SharedReIDGallery()
            print("[SM] 🌐 SharedReIDGallery yaratildi (cross-camera)", flush=True)
        except Exception as _ge:
            print(f"[SM] ⚠ reid_gallery error: {_ge}", flush=True)
            self.reid_gallery = None
        self.unknown_registry = UnknownRegistry()
        self.global_identity_cache = GlobalIdentityCache()
        self.room_manager = RoomManager()

        # ---------------- camera ----------------
        self.camera_manager = CameraManager(
            config=self.config,
            db=self.db,
            event_bus=self.event_bus,
        )

        # ---------------- features (ENDI face_engine MAVJUD) ----------------
        self.identity_manager = IdentityManager(
            config=self.config,
            db=self.db,
            db_writer=self.db_writer,
            event_bus=self.event_bus,
        )

        self.snapshot_service = SnapshotService(self.config)

        self.events_service = EventsService(
            config=self.config,
            db=self.db,
            db_writer=self.db_writer,
            event_bus=self.event_bus,
        )
        self.events_service.set_snapshot_service(self.snapshot_service)

        self.alerts_service = AlertsService(
            config=self.config,
            events_service=self.events_service,
        )

        self.analytics_service = AnalyticsService(
            config=self.config,
            db=self.db,
            db_writer=self.db_writer,
            identity_manager=self.identity_manager,
            events_service=self.events_service,
        )

        self.person_service = PersonService(
            config=self.config,
            db=self.db,
            face_engine=self.face_engine,       # ✅ ENDI MAVJUD
            db_writer=self.db_writer,
            identity_manager=self.identity_manager,)

        self.unknown_service = UnknownService(
            config=self.config,
            db=self.db,
            face_engine=self.face_engine,       # ✅ ENDI MAVJUD
            person_service=self.person_service,
        )

        self.enrollment_service = EnrollmentService(
            config=self.config,
            db=self.db,
            face_engine=self.face_engine,
            db_writer=self.db_writer,
)

        self.settings_service = SettingsService(
            config=self.config,
            db=self.db,
            db_writer=self.db_writer,
            camera_manager=self.camera_manager,
            detector=self.detector,
            face_engine=self.face_engine,       # ✅ ENDI MAVJUD
            alerts_service=self.alerts_service,
        )

        # ---------------- storage ----------------
        self.recording_service = RecordingService(self.config)

        self.cleanup_service = CleanupService(
            config=self.config,
            db=self.db,
        )

        self.export_service = ExportService(
            config=self.config,
            db=self.db,
        )

        # ---------------- performance ----------------
        self.performance_monitor = PerformanceMonitor(
            config=self.config,
            db_writer=self.db_writer,
            camera_manager=self.camera_manager,
            identity_manager=self.identity_manager,
        )

        # ---------------- AI workers ----------------
        self.ai_workers = {}

        self._connect_global_signals()

        log.info("ServiceManager initialized")

    
    # ---------------- global signals ----------------
    def _connect_global_signals(self):
        # camera status -> events + analytics
        try:
            self.camera_manager.status_changed.connect(
                self.events_service.camera_status_changed
            )

            self.camera_manager.status_changed.connect(
                self.analytics_service.set_camera_online
            )

            self.camera_manager.health_updated.connect(
                self._on_camera_health
            )

            self.camera_manager.frame_bgr_ready = None  # placeholder, workers connect below

        except Exception as e:
            log.error("camera_manager connect error: %s", e)

        # camera add/remove -> AI workers
        try:
            self.camera_manager.camera_added.connect(self._on_camera_added)
            self.camera_manager.camera_removed.connect(self._on_camera_removed)
        except Exception as e:
            log.error("camera add/remove connect error: %s", e)

        # enrollment -> events + person list refresh
        try:
            self.enrollment_service.person_registered.connect(
                self._on_enrollment_completed
            )
        except Exception as e:
            log.error("enrollment connect error: %s", e)

        # recording global enabled from settings
        try:
            self.recording_service.set_global_enabled(
                bool(self.config.get("storage.recordings_enabled", False))
            )
        except Exception as e:
            log.error("recording init error: %s", e)

    def _on_camera_health(self, camera_id: str, metrics: dict):
        try:
            self.analytics_service.set_camera_fps(camera_id, metrics.get("fps", 0.0))
        except Exception as e:
            log.error("_on_camera_health error: %s", e)

    def _on_enrollment_completed(self, result: dict):
        try:
            name = result.get("name", "")

            self.events_service.enrollment_completed(name)
            self.person_service.persons_changed.emit()

        except Exception as e:
            log.error("_on_enrollment_completed error: %s", e)

    # ---------------- AI workers ----------------
    def _create_tracker(self):
        return ByteTracker(
            track_buffer=int(self.config.get("ai.tracker.track_buffer", 30)),
            match_thresh=float(self.config.get("ai.tracker.match_thresh", 0.25)),
            high_thresh=float(self.config.get("ai.tracker.track_high_thresh", 0.35)),
            low_thresh=float(self.config.get("ai.tracker.track_low_thresh", 0.1)),
            new_track_thresh=0.3,
            reid_weight=0.35 if self.reid_engine.enabled and self.reid_engine.available else 0.0,
        )

    def _ensure_ai_worker(self, camera_id: str):
        if camera_id in self.ai_workers:
            return

        worker = self.camera_manager.get_worker(camera_id)

        if worker is None:
            return

        tracker = self._create_tracker()

        ai_worker = AIWorker(
            camera_id=camera_id,
            frame_buffer=worker.buffer,
            detector=self.detector,
            tracker=tracker,
            pose_engine=self.pose_engine,
            face_engine=self.face_engine,
            reid_engine=self.reid_engine,
            config=self.config,
            db_writer=self.db_writer,
)

        # MUHIM: Signal ulanish
        ai_worker.result_ready.connect(self.identity_manager.process_result)
        
        # DEBUG
        print(f"[DEBUG] AIWorker {camera_id} -> IdentityManager connected", flush=True)

        ai_worker.event_detected.connect(lambda cam_id, evt: self.events_service.publish_event(evt))

        try:
            worker.frame_bgr_ready.connect(self.recording_service.write_frame)
        except Exception as e:
            log.error("frame_bgr_ready connect error: %s", e)

        ai_worker.shared_gallery = getattr(self, "reid_gallery", None)
        ai_worker.start()

        self.ai_workers[camera_id] = ai_worker
        self.performance_monitor.set_ai_workers(self.ai_workers)

        log.info("AIWorker created: %s", camera_id)


    def _remove_ai_worker(self, camera_id: str):
        ai_worker = self.ai_workers.pop(camera_id, None)

        if ai_worker is not None:
            ai_worker.stop()
            ai_worker.wait(2000)

        self.performance_monitor.set_ai_workers(self.ai_workers)

        log.info("AIWorker removed: %s", camera_id)

    def _on_camera_added(self, cam: dict):
        try:
            self._ensure_ai_worker(cam.get("id"))
        except Exception as e:
            log.error("_on_camera_added error: %s", e)

    def _on_camera_removed(self, camera_id: str):
        try:
            self._remove_ai_worker(camera_id)
        except Exception as e:
            log.error("_on_camera_removed error: %s", e)

    # ---------------- start ----------------
    def get_next_global_id(self) -> int:
        """Thread-safe global ID generator. 1 dan boshlanadi."""
        with self._global_id_lock:
            self._global_id_counter += 1
            return self._global_id_counter

    def start(self):
        log.info("ServiceManager starting")

        # load cameras from DB/config
        self.camera_manager.load()

        # create AI workers for all cameras
        for camera_id in self.camera_manager.all_camera_ids():
            _cam = self.camera_manager.cameras.get(camera_id, {})
            if not _cam.get("online", False):
                print(f"[SM] ⏭ {camera_id} offline — AI worker YARATILMADI", flush=True)
                continue
            self._ensure_ai_worker(camera_id)

        # initial gallery reload
        try:
            self.face_engine.load_gallery()
        except Exception as e:
            log.error("face gallery load error: %s", e)

        self.ready.emit()

        log.info("ServiceManager started")

    # ---------------- shutdown ----------------
    def shutdown(self):
        log.info("ServiceManager shutting down")

        # stop AI workers
        for ai_worker in self.ai_workers.values():
            try:
                ai_worker.stop()
            except Exception:
                pass

        for ai_worker in self.ai_workers.values():
            try:
                ai_worker.wait(2000)
            except Exception:
                pass

        self.ai_workers = {}

        # stop cameras
        try:
            self.camera_manager.shutdown()
        except Exception as e:
            log.error("camera_manager shutdown error: %s", e)

        # stop services
        try:
            self.enrollment_service.shutdown()
        except Exception:
            pass

        try:
            self.identity_manager.shutdown()
        except Exception:
            pass

        try:
            self.analytics_service.shutdown()
        except Exception:
            pass

        try:
            self.cleanup_service.shutdown()
        except Exception:
            pass

        try:
            self.recording_service.shutdown()
        except Exception:
            pass

        try:
            self.performance_monitor.shutdown()
        except Exception:
            pass

        # stop DB writer
        try:
            self.db_writer.stop()
            self.db_writer.wait(3000)
        except Exception as e:
            log.error("db_writer shutdown error: %s", e)

        # close DB
        try:
            self.db.close()
        except Exception as e:
            log.error("db close error: %s", e)

        self.shutdown_finished.emit()

        log.info("ServiceManager stopped")