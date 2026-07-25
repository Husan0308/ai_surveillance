import io
import random
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap

from backend.core.logger import get_logger

log = get_logger("enrollment")


def _pixmap_to_png_bytes(pm):
    if pm is None or pm.isNull():
        return b""
    try:
        # ✅ Ishonchli usul: Qt ning o'zi PNG ga saqlaydi (numpy/asstring kerak emas)
        from PySide6.QtCore import QBuffer, QIODevice, QByteArray
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.WriteOnly)
        pm.save(buf, "PNG")
        buf.close()
        return bytes(ba)
    except Exception as e:
        log.error("_pixmap_to_png_bytes error: %s", e)
        return b""


def _jpg_bytes(bgr_frame):
    if bgr_frame is None:
        return b""
    try:
        ok, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else b""
    except Exception as e:
        log.error("_jpg_bytes error: %s", e)
        return b""


class EnrollmentService(QObject):
    status_changed = Signal(str)
    face_detected = Signal(bool)
    face_info_updated = Signal(bool, dict)
    quality_updated = Signal(float, dict)
    countdown_tick = Signal(int)
    capture_added = Signal(int, int)
    duplicate_detected = Signal(int, str, float)
    person_registered = Signal(dict)
    finished = Signal(bool, str)

    capture_progress = Signal(int, int)
    thumbnail_captured = Signal(object)
    embedding_ready = Signal(bool, object)
    shutter_requested = Signal()
    flash_requested = Signal()
    frame_preview = Signal(object)

    def __init__(self, config, db, face_engine=None, db_writer=None):
        super().__init__()
        self.config = config
        self.db = db
        self.face_engine = face_engine
        self.db_writer = db_writer

        if self.db is None:
            raise RuntimeError("EnrollmentService: db=None! ServiceManager must pass db=self.db")

        self.state = "idle"
        self.current_embedding = None
        self.current_embedding_score = 0.0
        self.captures_pix = []
        self.captures_bgr = []

        self.images_count = int(config.get("enrollment.images_count", 10))
        self.min_quality = float(config.get("enrollment.min_face_quality", 45))
        self.countdown_seconds = int(config.get("enrollment.countdown_seconds", 3))
        self.duplicate_check = bool(config.get("enrollment.duplicate_check", True))
        self.duplicate_threshold = float(config.get("face.match_threshold", 0.58))

        self._detect_timer = QTimer(self)
        self._detect_timer.setSingleShot(True)
        self._detect_timer.timeout.connect(self._do_capture)

        self._count_timer = QTimer(self)
        self._count_timer.setInterval(1000)
        self._count_timer.timeout.connect(self._countdown_step)
        self.countdown_value = 0

        log.info("EnrollmentService initialized (db=%s)", getattr(self.db, "path", "?"))

    def start_session(self):
        self.state = "detecting"
        self.captures_pix.clear()
        self.captures_bgr.clear()
        self.current_embedding = None
        self.current_embedding_score = 0.0
        self.status_changed.emit("🟢 Searching for face…")
        self._detect_timer.start(200)

    def begin_capture_sequence(self):
        """ui_patches.py dan chaqiriladi — start_session bilan bir xil"""
        self.start_session()

    def stop_session(self):
        self._detect_timer.stop()
        self._count_timer.stop()
        self.state = "idle"
        self.status_changed.emit("⏹ Session stopped")

    def _do_capture(self):
        if self.state != "detecting":
            return
        self.state = "countdown"
        self.countdown_value = self.countdown_seconds
        self._countdown_step()
        self._count_timer.start()

    def _countdown_step(self):
        if self.countdown_value > 0:
            self.countdown_tick.emit(self.countdown_value)
            self.status_changed.emit(f"⏱ Hold still… {self.countdown_value}")
            self.countdown_value -= 1
            return
        self._count_timer.stop()
        self.status_changed.emit("📸 Capturing…")
        self.state = "processing"

    def add_capture(self, pixmap, bgr_frame, embedding, score):
        self.captures_pix.append(pixmap)
        self.captures_bgr.append(bgr_frame)
        self.current_embedding = embedding
        self.current_embedding_score = float(score)
        n = len(self.captures_pix)

        # ✅ Eski signal
        self.capture_added.emit(n, self.images_count)
        # ✅ Yangi signallar (ui_patches uchun)
        self.capture_progress.emit(n, self.images_count)
        self.thumbnail_captured.emit(pixmap)

        self.status_changed.emit(f"📷 Captured {n}/{self.images_count}")
        if n >= self.images_count:
            self.state = "ready"
            self.status_changed.emit("✅ Ready to register")
            self.embedding_ready.emit(True, embedding)
            self.finished.emit(True, "Captures complete")

    def set_embedding(self, embedding, score):
        self.current_embedding = embedding
        self.current_embedding_score = float(score)
        self.state = "ready"
        self.status_changed.emit("✅ Embedding ready — press Register")
        # ✅ Yangi signal (ui_patches uchun)
        self.embedding_ready.emit(True, embedding)

    def register_person(self, name, department, employee_id=""):
        name = (name or "").strip()
        if not name:
            self.finished.emit(False, "Name required")
            return False

        if self.state != "ready" or self.current_embedding is None:
            self.finished.emit(False, "Not ready")
            return False

        if not employee_id:
            employee_id = f"EMP-{random.randint(1000, 9999)}"

        avatar_pm = self.captures_pix[0] if self.captures_pix else QPixmap()
        avatar_bytes = _pixmap_to_png_bytes(avatar_pm)

        # ✅ TO'G'RIDAN-TO'G'RI SQL YOZISH
        try:
            from datetime import datetime
            now = datetime.now().isoformat()

            sql = """
                INSERT INTO persons (
                    name, department, employee_id, status, avatar,
                    created_at, updated_at, rec_count, stay_total
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)
            """
            params = (name, department, employee_id, "Active", avatar_bytes, now, now)

            with self.db.lock:
                cursor = self.db.conn.execute(sql, params)
                person_id = cursor.lastrowid
                self.db.conn.commit()

            if not person_id:
                raise RuntimeError(f"Invalid person_id: {person_id}")

        except Exception as e:
            log.error("register_person DB error: %s", e)
            self.finished.emit(False, f"DB Error: {e}")
            return False

        try:
            emb_bytes = np.asarray(self.current_embedding, dtype=np.float32).tobytes()
            img_bytes = _jpg_bytes(self.captures_bgr[0]) if self.captures_bgr else b""
            with self.db.lock:
                self.db.conn.execute(
                    "INSERT INTO face_embeddings (person_id, embedding, image, quality, created_at) VALUES (?,?,?,?,?)",
                    (person_id, emb_bytes, img_bytes, float(self.current_embedding_score), now),
                )
                self.db.conn.commit()
        except Exception as e:
            log.error("register_person embedding error: %s", e)

        try:
            if self.face_engine:
                self.face_engine.add_to_gallery(person_id, name, self.current_embedding)
        except Exception as e:
            log.error("register_person gallery error: %s", e)

        result = {
            "id": person_id, "name": name, "department": department,
            "employee_id": employee_id, "status": "Active",
            "avatar": avatar_pm, "embedding_score": self.current_embedding_score,
        }
        self.person_registered.emit(result)
        self.status_changed.emit(f"✅ {name} registered")
        self.finished.emit(True, f"{name} registered")
        self._reset_after_register()
        return True

    def _reset_after_register(self):
        self.state = "idle"
        self.current_embedding = None
        self.current_embedding_score = 0.0
        self.captures_pix.clear()
        self.captures_bgr.clear()