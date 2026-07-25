import cv2
from datetime import datetime

from PySide6.QtCore import QObject, Signal, QRectF
from PySide6.QtGui import QImage, QPixmap

from backend.core.logger import get_logger

from ui import make_avatar

log = get_logger("bridge.system")


# ============================ HELPERS ================================
def bgr_to_qimage(bgr):
    if bgr is None:
        return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def parse_time(value):
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


# ============================ PERSON UI ==============================
class RealPersonUI:
    """
    UI VideoSurface._draw_ai kutadigan person ob’ekti.
    """

    def __init__(self, detected_person, frame_w=640, frame_h=360):
        self.track_id = detected_person.track_id
        self.known = bool(detected_person.known)
        self.name = detected_person.name or f"Person {detected_person.track_id}"

        self.conf = float(detected_person.conf)
        self.face_conf = float(detected_person.face_conf)

        # UI dwell time: ps.age / 25.0
        # DetectedPerson.age seconds, shuning uchun frame ga aylantiramiz
        self.age = int(float(detected_person.age) * 25.0)

        self.box = list(detected_person.box)
        self.frame_w = float(frame_w)
        self.frame_h = float(frame_h)

        self.ankle = detected_person.ankle

        self.overstay_alerted = False

    def bbox(self, W, H):
        x1, y1, x2, y2 = self.box[:4][:4][:4]

        sx = W / self.frame_w if self.frame_w else 1.0
        sy = H / self.frame_h if self.frame_h else 1.0

        return QRectF(
            x1 * sx,
            y1 * sy,
            (x2 - x1) * sx,
            (y2 - y1) * sy,
        )


# ============================ ENROLLMENT FACE PROXY ==================
class EnrollmentFaceProxyUI:
    """
    UI EnrollmentPage._draw_face kutadigan proxy.

    Face bbox dan artificial body bbox yasaydi,
    chunki UI face box ni body proportions orqali chizadi.
    """

    def __init__(self, face_info, frame_w=640, frame_h=360):
        self.track_id = 0
        self.known = False
        self.name = "Face"

        self.face_conf = float(face_info.get("quality_score", 0.0)) / 100.0
        self.conf = self.face_conf
        self.age = 0
        self.overstay_alerted = False

        bbox = face_info.get("bbox", [0, 0, 1, 1])

        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        face_h = max(1.0, y2 - y1)

        ph = face_h / 0.399
        hr = ph * 0.105
        top = cy - 1.5 * hr
        pw = ph * 0.36

        self.body_box = (cx - pw / 2.0, top, pw, ph)

        self.frame_w = float(frame_w)
        self.frame_h = float(frame_h)

    def bbox(self, W, H):
        x, y, w, h = self.body_box

        sx = W / self.frame_w if self.frame_w else 1.0
        sy = H / self.frame_h if self.frame_h else 1.0

        return QRectF(x * sx, y * sy, w * sx, h * sy)


# ============================ PERSON RECORD UI =======================
class PersonRecordUI:
    """
    UI PersonManagementPage va ProfileDialog kutadigan record.
    """

    def __init__(self, data: dict, person_service=None):
        self.db_id = data.get("id")

        self.name = data.get("name", "")
        self.dept = data.get("department", "")
        self.emp_id = data.get("employee_id", "")
        self.status = data.get("status", "Active")

        self.last_seen = parse_time(data.get("last_seen")) or datetime.now()
        self.rec_count = int(data.get("rec_count", 0) or 0)

        # ✅ AVATAR YUKLASH - ishonchli usul
        avatar_data = data.get("avatar")
        
        if avatar_data and len(avatar_data) > 0:
            try:
                self.avatar = QPixmap()
                loaded = self.avatar.loadFromData(avatar_data)
                
                if loaded and not self.avatar.isNull():
                    print(f"[PersonRecordUI] {self.name}: avatar loaded {self.avatar.width()}x{self.avatar.height()}", flush=True)
                else:
                    print(f"[PersonRecordUI] {self.name}: avatar load failed, using default", flush=True)
                    self.avatar = make_avatar(self.name, 96)
            except Exception as e:
                print(f"[PersonRecordUI] {self.name}: avatar error {e}, using default", flush=True)
                self.avatar = make_avatar(self.name, 96)
        else:
            print(f"[PersonRecordUI] {self.name}: no avatar data, using default", flush=True)
            self.avatar = make_avatar(self.name, 96)

        self.timeline = [0] * 24
        self.visited = []
        self.stay_total = 0
        self.history = []

        if person_service is not None:
            self._load_profile(person_service)
            
    def _load_profile(self, person_service):
        try:
            profile = person_service.get_full_profile(self.db_id)

            if profile is None:
                return

            self.timeline = profile.get("timeline", [0] * 24)

            visits = []

            for v in profile.get("visit_history", []):
                entered = parse_time(v.get("entered_at"))
                entered_str = entered.strftime("%H:%M") if entered else "--:--"

                duration_min = int(v.get("duration_sec", 0) or 0) // 60

                visits.append((
                    v.get("camera_id", ""),
                    entered_str,
                    duration_min,
                ))

            self.visited = visits
            self.stay_total = int(profile.get("total_stay_sec", 0) or 0) // 60

            history = []

            for e in profile.get("events", [])[:12]:
                t = parse_time(e.get("time")) or datetime.now()
                txt = f"{e.get('type', '')} at {e.get('camera_id', '')}"
                history.append((t, txt))

            self.history = history

        except Exception as e:
            log.error("PersonRecordUI profile load error: %s", e)


# ============================ REAL CAMERA SIM ========================
class RealCameraSim(QObject):
    """
    UI CameraSim contract.
    """

    def __init__(self, camera_id, name, location, service_manager):
        super().__init__()

        self.id = camera_id
        self.name = name
        self.location = location

        self.sm = service_manager

        self.surfaces = []

        self.ai_on = True
        self.heat_on = False
        self.recording = False

        self.fps = 0.0
        self.res = "—"
        self.latency = 0.0
        self.packet_loss = 0.0
        self.infer_ms = 0.0

        self.frame = None
        self.people = []

        self.zone_y = 0.68

        self._online = False

    # ---------------- online ----------------
    @property
    def online(self):
        return self._online

    @online.setter
    def online(self, value):
        value = bool(value)

        if value == self._online:
            return

        self._online = value

        try:
            if value:
                self.sm.camera_manager.start_camera(self.id)
            else:
                self.sm.camera_manager.stop_camera(self.id)
        except Exception as e:
            log.error("RealCameraSim online setter error: %s", e)

    def set_online_status(self, online: bool):
        self._online = bool(online)

    # ---------------- backend updates ----------------
    def update_from_ai_result(self, camera_id, result):
        if camera_id != self.id:
            return

        try:
            if result.frame is not None:
                self.frame = bgr_to_qimage(result.frame)

            frame_h, frame_w = 360, 640

            if result.frame is not None:
                frame_h, frame_w = result.frame.shape[:2]

            self.people = [
                RealPersonUI(p, frame_w, frame_h)
                for p in result.persons
            ]

            self.infer_ms = float(result.infer_ms)

            # DEBUG
            if len(self.people) > 0:
                print(f"[DEBUG] {self.id}: {len(self.people)} persons, frame={frame_w}x{frame_h}", flush=True)

        except Exception as e:
            log.error("update_from_ai_result error: %s", e)
    def update_health(self, camera_id, metrics):
        if camera_id != self.id:
            return

        try:
            self.fps = float(metrics.get("fps", 0.0))
            self.latency = float(metrics.get("latency_ms", 0.0))
            self.packet_loss = float(metrics.get("packet_loss", 0.0))

            online = bool(metrics.get("online", self._online))
            self._online = online

        except Exception as e:
            log.error("update_health error: %s", e)

    # ---------------- UI contract ----------------
    def step(self):
        # Backend threadlar real ishni qiladi.
        # UI tick faqat surface update qiladi.
        pass

    def render(self):
        # UI camera menu / settings render chaqiradi.
        # Real camera worker start/stop online setter orqali qilinadi.
        pass

    def spawn_person(self, initial=False):
        return None

    def heat_image(self):
        try:
            return self.sm.identity_manager.get_heatmap_image(self.id)
        except Exception as e:
            log.error("heat_image error: %s", e)
            return QImage()

    @property
    def conn_quality(self):
        try:
            worker = self.sm.camera_manager.get_worker(self.id)

            if worker is None:
                return 0

            return worker.health.conn_quality

        except Exception:
            return 0

    @property
    def known_count(self):
        return sum(1 for p in self.people if getattr(p, "known", False))

    @property
    def unknown_count(self):
        return max(0, len(self.people) - self.known_count)


# ============================ REAL ENROLLMENT SIM ====================
class RealEnrollmentSim(RealCameraSim):
    def __init__(self, service_manager):
        super().__init__(
            "CAM-EN",
            "Enrollment Station",
            "HQ — Security Office",
            service_manager,
        )

        self._online = False
        self.ai_on = False
        self._cap = None
        self._face_count = 0

        from PySide6.QtCore import QTimer
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(33)
        self._frame_timer.timeout.connect(self._update_frame)

        try:
            self.sm.enrollment_service.face_info_updated.connect(self._on_face_info)
        except Exception as e:
            log.error("RealEnrollmentSim connect error: %s", e)

    @property
    def online(self):
        return self._online

    @online.setter
    def online(self, value):
        value = bool(value)
        if value == self._online:
            return
        self._online = value
        if value:
            self._open_camera()
            self._frame_timer.start()
            log.info("Enrollment kamera YOQILDI ✅")
        else:
            self._frame_timer.stop()
            self._close_camera()
            log.info("Enrollment kamera O'CHIRILDI ⏹")

    def set_online_status(self, online: bool):
        self.online = online

    def _open_camera(self):
        import cv2
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(0)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # ✅ 640 emas
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # ✅ 480 emas
                # ✅ Autofocus va exposure yaxshilash
                self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                log.info("Webkamera ochildi 1280x720 ✅")
            else:
                log.error("Webkamera ochilmadi ❌")

    def _close_camera(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.frame = None
        self.people = []

    def _update_frame(self):
        if self._cap is None or not self._cap.isOpened():
            return

        ret, frame = self._cap.read()
        if not ret:
            return

        self.frame = bgr_to_qimage(frame)

        # Har 10-frame da (~330ms) face detection
        self._face_count += 1
        if self._face_count % 10 == 0:
            try:
                # ✅ need_embedding=False → burilgan yuzda ham BOX chiziladi
                faces = self.sm.face_engine.detect(frame, need_embedding=False)

                if faces and len(faces) > 0:
                    face = faces[0]
                    face["quality_score"] = face.get("score", 0.0) * 100.0

                    fh, fw = frame.shape[:2]

                    # ✅ Face bbox ni saqlash (capture_one portret crop uchun)
                    self._face_bbox = face.get("bbox")

                    self.people = [EnrollmentFaceProxyUI(face, fw, fh)]

                    # Embedding bor bo'lsa saqlash (ixtiyoriy)
                    emb = face.get("embedding")
                    if emb is not None:
                        self.sm.enrollment_service.current_embedding = emb
                        self.sm.enrollment_service.current_embedding_score = face.get("score", 0.0)
                else:
                    self.people = []
                    self._face_bbox = None

            except Exception as e:
                log.error("Enrollment face detect error: %s", e)


    def _on_face_info(self, has_face, face_info):
        try:
            if has_face and face_info is not None:
                self.people = [EnrollmentFaceProxyUI(face_info)]
            else:
                self.people = []
        except Exception as e:
            log.error("RealEnrollmentSim face info error: %s", e)

    def stop(self):
        if hasattr(self, '_frame_timer'):
            self._frame_timer.stop()
        self._close_camera()
        log.info("Enrollment kamera to'liq yopildi ⏹")

class RealSystem(QObject):
    """
    UI MainWindow kutadigan System contract.
    """

    new_event = Signal(dict)
    people_changed = Signal()

    def __init__(self, service_manager):
        super().__init__()

        self.sm = service_manager

        # UI settings format
        self.settings = self._build_settings()

        # real-time system stats
        self._gpu = 0.0
        self._cpu = 0.0
        self._ram = 0.0

        # UI analytics helpers
        self.visitors = {}
        self.usage = {}
        self.peak = [0] * 24

        # events UI format
        self.events = []

        # people UI format
        self.people = []

        # camera sims
        self.sims = []

        # enrollment sim
        self.enroll_sim = RealEnrollmentSim(service_manager)

        self._build_camera_sims()
        self._reload_events()
        self._reload_people()
        self._connect_signals()

        log.info("RealSystem bridge initialized")

    # ---------------- settings ----------------
    def _build_settings(self):
        s = self.sm.settings_service.settings

        return {
            # UI PasswordDialog plain compare uchun vaqtincha.
            # Step 13’da verify_password bilan almashtiriladi.
            "password": "admin",
            "unlocked": False,

            "det_conf": float(s.get("det_conf", 0.45)),
            "face_th": float(s.get("face_threshold", 0.58)),

            "model": self.sm.config.get("ai.detector.model", "yolov8n.pt"),
            "retention": int(s.get("events_retention_days", 30)),
            "sound": bool(s.get("sound_enabled", True)),
        }

    # ---------------- camera sims ----------------
    def _build_camera_sims(self):
        self.sims = []

        for camera_id, cam in self.sm.camera_manager.cameras.items():
            sim = RealCameraSim(
                camera_id=cam.get("id"),
                name=cam.get("name", cam.get("id")),
                location=cam.get("location", ""),
                service_manager=self.sm,
            )

            sim.set_online_status(bool(cam.get("online", False)))
            sim.recording = bool(cam.get("recording_enabled", False))

            # AI result -> camera sim
            ai_worker = self.sm.ai_workers.get(camera_id)

            if ai_worker is not None:
                try:
                    ai_worker.result_ready.connect(sim.update_from_ai_result)
                except Exception as e:
                    log.error("ai_worker connect error: %s", e)

            # camera health -> camera sim
            worker = self.sm.camera_manager.get_worker(camera_id)

            if worker is not None:
                try:
                    worker.health_updated.connect(sim.update_health)
                except Exception as e:
                    log.error("worker health connect error: %s", e)

            self.sims.append(sim)

        # camera status -> sims
        try:
            self.sm.camera_manager.status_changed.connect(self._on_camera_status)
        except Exception as e:
            log.error("camera_manager status connect error: %s", e)

    def _on_camera_status(self, camera_id, online):
        for sim in self.sims:
            if sim.id == camera_id:
                sim.set_online_status(bool(online))
                break

    # ---------------- events ----------------
    def _event_to_ui(self, e: dict) -> dict:
        return {
            "time": e.get("time") or datetime.now(),
            "cam": e.get("camera_id", "SYS"),
            "person": e.get("person_name", ""),
            "type": e.get("type", "system"),
            "conf": float(e.get("confidence", 0.0) or 0.0),
            "level": e.get("level", "info"),
            "ack": bool(e.get("ack", False)),
            "snapshot_path": e.get("snapshot_path"),
            "_src": e,
        }

    def _reload_events(self):
        try:
            self.events = [
                self._event_to_ui(e)
                for e in self.sm.events_service.events
            ]
        except Exception as e:
            log.error("_reload_events error: %s", e)
            self.events = []

    def _on_event_added(self, e: dict):
        try:
            # ✅ Odam tanilganda last_seen + rec_count yangilanadi
            if e.get("type") == "person_recognized":
                pid = e.get("person_id")
                if pid is not None:
                    try:
                        self.sm.db.update_person_last_seen(pid, 1)
                    except Exception as ex:
                        log.error("update_person_last_seen error: %s", ex)

            ui_e = self._event_to_ui(e)
            self.new_event.emit(ui_e)
        except Exception as ex:
            log.error("_on_event_added error: %s", ex)

    def push_event(self, e: dict, silent=False):
        """
        UI push_event format:
            {type, level, cam, person, conf}
        """

        try:
            ev = {
                "type": e.get("type", "system"),
                "level": e.get("level", "info"),
                "camera_id": e.get("cam") or e.get("camera_id", "SYS"),
                "person_name": e.get("person") or e.get("person_name", ""),
                "confidence": float(e.get("conf", e.get("confidence", 0.0)) or 0.0),
                "ack": bool(e.get("ack", False)),
                "snapshot_path": e.get("snapshot_path"),
            }

            self.sm.events_service.publish_event(ev)

        except Exception as ex:
            log.error("push_event error: %s", ex)

    # ---------------- people ----------------
    def _reload_people(self):
        print(f"[RealSystem] _reload_people called", flush=True)
        try:
            rows = self.sm.person_service.load_persons()
            print(f"[RealSystem] Loaded {len(rows)} persons from DB", flush=True)

            self.people = [
                PersonRecordUI(row, self.sm.person_service)
                for row in rows
            ]
            
            print(f"[RealSystem] people list updated: {len(self.people)}", flush=True)
            self.people_changed.emit()

        except Exception as e:
            print(f"[RealSystem] _reload_people error: {e}", flush=True)
            log.error("_reload_people error: %s", e)
            self.people = []

    # ---------------- analytics ----------------
    def _on_analytics(self, data: dict):
        try:
            self._gpu = float(data.get("gpu", 0.0))
            self._cpu = float(data.get("cpu", 0.0))
            self._ram = float(data.get("ram", 0.0))

            self.peak = data.get("peak_hour", [0] * 24)

            # visitors = detection totals per camera
            visitors = {}

            for camera_id, state in self.sm.identity_manager.states.items():
                visitors[camera_id] = state.detections_total

            self.visitors = visitors

            # usage minutes
            usage = {}

            for camera_id, seconds in self.sm.analytics_service.utilization_seconds.items():
                usage[camera_id] = int(seconds // 60)

            self.usage = usage

        except Exception as e:
            log.error("_on_analytics error: %s", e)

    # ---------------- connect ----------------
    def _connect_signals(self):
        try:
            self.sm.events_service.event_added.connect(self._on_event_added)
        except Exception as e:
            log.error("events_service connect error: %s", e)

        try:
            self.sm.analytics_service.analytics_updated.connect(self._on_analytics)
        except Exception as e:
            log.error("analytics_service connect error: %s", e)

        try:
            self.sm.person_service.persons_changed.connect(self._reload_people)
        except Exception as e:
            log.error("person_service connect error: %s", e)

        try:
            self.sm.settings_service.settings_changed.connect(self._on_settings_changed)
        except Exception as e:
            log.error("settings_service connect error: %s", e)

    def _on_settings_changed(self, public_settings: dict):
        try:
            self.settings["det_conf"] = float(public_settings.get("det_conf", 0.45))
            self.settings["face_th"] = float(public_settings.get("face_threshold", 0.58))
            self.settings["sound"] = bool(public_settings.get("sound_enabled", True))
            self.settings["retention"] = int(public_settings.get("events_retention_days", 30))
        except Exception as e:
            log.error("_on_settings_changed error: %s", e)

    # ---------------- UI helpers ----------------
    def sim_by_id(self, camera_id: str):
        for sim in self.sims:
            if sim.id == camera_id:
                return sim

        if self.enroll_sim.id == camera_id:
            return self.enroll_sim

        return None

    @property
    def cams_online(self):
        return sum(1 for sim in self.sims if sim.online)

    @property
    def gpu(self):
        return self._gpu

    @gpu.setter
    def gpu(self, value):
        pass

    @property
    def cpu(self):
        return self._cpu

    @cpu.setter
    def cpu(self, value):
        pass

    @property
    def ram(self):
        return self._ram

    @ram.setter
    def ram(self, value):
        pass


# ============================ FACTORY ================================
def build_real_system(service_manager) -> RealSystem:
    return RealSystem(service_manager)