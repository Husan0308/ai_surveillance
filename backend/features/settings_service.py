import os
import time
import shutil
import hashlib
from datetime import datetime

from PySide6.QtCore import QObject, Signal

from backend.core.logger import get_logger

log = get_logger("features.settings")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def hash_password(password: str) -> str:
    return hashlib.sha256(str(password).encode("utf-8")).hexdigest()


class SettingsService(QObject):
    """
    Settings service.

    - password
    - cameras add/edit/delete
    - connection test
    - AI thresholds
    - alerts
    - database backup/vacuum
    - storage info
    - live apply
    """

    settings_changed = Signal(dict)
    camera_saved = Signal(dict)
    camera_deleted = Signal(str)
    password_changed = Signal()
    message = Signal(str, str)   # text, level: info/success/error

    def __init__(
        self,
        config,
        db,
        db_writer=None,
        camera_manager=None,
        detector=None,
        face_engine=None,
        alerts_service=None,
    ):
        super().__init__()

        self.config = config
        self.db = db
        self.db_writer = db_writer

        self.camera_manager = camera_manager
        self.detector = detector
        self.face_engine = face_engine
        self.alerts_service = alerts_service

        self.settings = {}

        self._load_defaults()
        self._load_from_db()
        self._ensure_password()

        self.apply_live()

        log.info("SettingsService started")

    # ---------------- defaults ----------------
    def _load_defaults(self):
        self.settings = {
            "password_hash": hash_password("admin"),
            "login_username": self.config.get("security.login_username", "admin"),

            "det_conf": float(self.config.get("ai.detector.conf", 0.45)),
            "face_threshold": float(self.config.get("ai.face.match_threshold", 0.58)),
            "ai_fps": int(self.config.get("ai.ai_fps", 8)),
            "face_interval_frames": int(self.config.get("ai.face_interval_frames", 6)),

            "sound_enabled": bool(self.config.get("alerts.sound_enabled", True)),
            "sound_events": self.config.get(
                "alerts.sound_events",
                ["intrusion", "camera_offline", "overstay", "error"],
            ),

            "events_retention_days": int(self.config.get("events.retention_days", 30)),

            "recordings_enabled": bool(self.config.get("storage.recordings_enabled", False)),
            "storage_retention_days": int(self.config.get("storage.retention_days", 14)),
            "max_disk_gb": int(self.config.get("storage.max_disk_gb", 500)),

            "auto_lock_minutes": int(self.config.get("security.auto_lock_minutes", 15)),
        }

    def _load_from_db(self):
        try:
            saved = self.db.get_settings()

            for k, v in saved.items():
                self.settings[k] = v

        except Exception as e:
            log.error("settings load_from_db error: %s", e)

    def _ensure_password(self):
        if not self.settings.get("password_hash"):
            self.settings["password_hash"] = hash_password("admin")
            self.db.save_setting("password_hash", self.settings["password_hash"])

    # ---------------- password ----------------
    def verify_password(self, password: str) -> bool:
        return self.settings.get("password_hash") == hash_password(password)

    def change_password(self, old_password: str, new_password: str) -> bool:
        if not self.verify_password(old_password):
            self.message.emit("Incorrect current password", "error")
            return False

        if not new_password or len(new_password) < 3:
            self.message.emit("New password is too short", "error")
            return False

        self.settings["password_hash"] = hash_password(new_password)
        self.db.save_setting("password_hash", self.settings["password_hash"])

        self.password_changed.emit()
        self.message.emit("Password changed", "success")

        log.info("Password changed")

        return True

    # ---------------- settings ----------------
    def get_public_settings(self) -> dict:
        out = dict(self.settings)
        out.pop("password_hash", None)
        return out

    def save_settings(self, data: dict):
        if not isinstance(data, dict):
            return

        # password hash must not be overwritten directly
        data.pop("password_hash", None)

        for key, value in data.items():
            self.settings[key] = value

            try:
                self.db.save_setting(key, value)
            except Exception as e:
                log.error("save_setting error: %s %s", key, e)

        # also update config file for important keys
        try:
            self.config.set("ai.detector.conf", float(self.settings.get("det_conf", 0.45)))
            self.config.set("ai.face.match_threshold", float(self.settings.get("face_threshold", 0.58)))
            self.config.set("ai.ai_fps", int(self.settings.get("ai_fps", 8)))
            self.config.set("ai.face_interval_frames", int(self.settings.get("face_interval_frames", 6)))

            self.config.set("alerts.sound_enabled", bool(self.settings.get("sound_enabled", True)))
            self.config.set("alerts.sound_events", self.settings.get("sound_events", []))

            self.config.set("events.retention_days", int(self.settings.get("events_retention_days", 30)))

            self.config.set("storage.recordings_enabled", bool(self.settings.get("recordings_enabled", False)))
            self.config.set("storage.retention_days", int(self.settings.get("storage_retention_days", 14)))
            self.config.set("storage.max_disk_gb", int(self.settings.get("max_disk_gb", 500)))

            self.config.set("security.auto_lock_minutes", int(self.settings.get("auto_lock_minutes", 15)))

            self.config.save()
        except Exception as e:
            log.error("config save error: %s", e)

        self.apply_live()

        self.settings_changed.emit(self.get_public_settings())
        self.message.emit("Settings saved", "success")

        log.info("Settings saved")

    def apply_live(self):
        """
        O'zgarishlarni darhol backend modullarga qo'llaydi.
        """

        try:
            if self.detector is not None:
                self.detector.conf = float(self.settings.get("det_conf", 0.45))
        except Exception as e:
            log.error("apply detector conf error: %s", e)

        try:
            if self.face_engine is not None:
                self.face_engine.threshold = float(self.settings.get("face_threshold", 0.58))
        except Exception as e:
            log.error("apply face threshold error: %s", e)

        try:
            if self.alerts_service is not None:
                self.alerts_service.set_sound_enabled(
                    bool(self.settings.get("sound_enabled", True))
                )
                self.alerts_service.set_sound_events(
                    self.settings.get("sound_events", [])
                )
        except Exception as e:
            log.error("apply alerts settings error: %s", e)

    # ---------------- cameras ----------------
    def get_cameras(self):
        if self.camera_manager is not None:
            return list(self.camera_manager.cameras.values())

        try:
            return self.db.get_camera_configs()
        except Exception:
            return []

    def _next_camera_id(self):
        cams = self.get_cameras()

        max_num = 0

        for cam in cams:
            cid = str(cam.get("id", ""))

            if cid.startswith("CAM-"):
                try:
                    num = int(cid.split("-", 1)[1])
                    max_num = max(max_num, num)
                except Exception:
                    pass

        return f"CAM-{max_num + 1:02d}"

    def add_camera(self, cam: dict):
        if self.camera_manager is None:
            self.message.emit("Camera manager unavailable", "error")
            return False

        cam = dict(cam)

        if not cam.get("id"):
            cam["id"] = self._next_camera_id()

        if not cam.get("source"):
            self.message.emit("Camera source is required", "error")
            return False

        cam.setdefault("name", cam["id"])
        cam.setdefault("location", "")
        cam.setdefault("online", False)
        cam.setdefault("ai_enabled", True)
        cam.setdefault("heatmap_enabled", False)
        cam.setdefault("recording_enabled", False)
        cam.setdefault("zone_enabled", False)
        cam.setdefault("overstay_enabled", False)
        cam.setdefault("resolution", "1920x1080")
        cam.setdefault("fps", 25)
        cam.setdefault("reconnect_interval", 10)
        cam.setdefault("connection_timeout", 5)
        cam.setdefault("latency_warn_ms", 200)
        cam.setdefault("packet_loss_warn_percent", 2.0)

        ok = self.camera_manager.add_camera(cam, persist=True)

        if ok:
            self.camera_saved.emit(cam)
            self.message.emit(f"Camera {cam['id']} added", "success")
            log.info("Camera added: %s", cam["id"])
        else:
            self.message.emit(f"Failed to add camera {cam['id']}", "error")

        return ok

    def update_camera(self, cam: dict):
        if self.camera_manager is None:
            self.message.emit("Camera manager unavailable", "error")
            return False

        cam = dict(cam)

        if not cam.get("id"):
            self.message.emit("Camera id is required", "error")
            return False

        ok = self.camera_manager.update_camera(cam)

        if ok:
            self.camera_saved.emit(cam)
            self.message.emit(f"Camera {cam['id']} updated", "success")
            log.info("Camera updated: %s", cam["id"])
        else:
            self.message.emit(f"Failed to update camera {cam['id']}", "error")

        return ok

    def delete_camera(self, camera_id: str):
        if self.camera_manager is None:
            self.message.emit("Camera manager unavailable", "error")
            return False

        self.camera_manager.delete_camera(camera_id)

        self.camera_deleted.emit(camera_id)
        self.message.emit(f"Camera {camera_id} deleted", "success")

        log.info("Camera deleted: %s", camera_id)

        return True

    def test_camera(self, source, username=None, password=None, timeout: int = 5):
        if self.camera_manager is None:
            return {
                "ok": False,
                "message": "Camera manager unavailable",
            }

        result = self.camera_manager.test_connection(
            source=source,
            username=username,
            password=password,
            timeout=timeout,
        )

        if result.get("ok"):
            self.message.emit("Camera connection OK", "success")
        else:
            self.message.emit(result.get("message", "Connection failed"), "error")

        return result

    # ---------------- database ----------------
    def backup_database(self):
        try:
            backups_dir = self.config.get("storage.backups_dir", "backups")

            if not os.path.isabs(backups_dir):
                backups_dir = os.path.join(BASE_DIR, backups_dir)

            os.makedirs(backups_dir, exist_ok=True)

            src = self.db.path

            filename = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            dst = os.path.join(backups_dir, filename)

            shutil.copy2(src, dst)

            self.message.emit(f"Database backup created: {filename}", "success")
            log.info("Database backup: %s", dst)

            return dst

        except Exception as e:
            log.error("backup_database error: %s", e)
            self.message.emit(f"Backup failed: {e}", "error")
            return None

    def vacuum_database(self):
        try:
            with self.db.lock:
                self.db.conn.execute("VACUUM")

            self.message.emit("Database optimized", "success")
            log.info("Database vacuum done")

            return True

        except Exception as e:
            log.error("vacuum_database error: %s", e)
            self.message.emit(f"Vacuum failed: {e}", "error")
            return False

    # ---------------- storage ----------------
    def _dir_size(self, path: str) -> int:
        total = 0

        if not os.path.exists(path):
            return 0

        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)

                try:
                    total += os.path.getsize(fp)
                except Exception:
                    pass

        return total

    def get_storage_info(self) -> dict:
        snapshots_dir = self.config.get("storage.snapshots_dir", "snapshots")
        recordings_dir = self.config.get("storage.recordings_dir", "recordings")
        exports_dir = self.config.get("storage.exports_dir", "exports")
        backups_dir = self.config.get("storage.backups_dir", "backups")
        logs_dir = self.config.get("logging.dir", "logs")
        data_dir = os.path.dirname(self.db.path)

        def abs_path(p):
            if os.path.isabs(p):
                return p
            return os.path.join(BASE_DIR, p)

        sizes = {
            "snapshots": self._dir_size(abs_path(snapshots_dir)),
            "recordings": self._dir_size(abs_path(recordings_dir)),
            "exports": self._dir_size(abs_path(exports_dir)),
            "backups": self._dir_size(abs_path(backups_dir)),
            "logs": self._dir_size(abs_path(logs_dir)),
            "database": self._dir_size(data_dir),
        }

        total_used = sum(sizes.values())

        try:
            usage = shutil.disk_usage(BASE_DIR)
            disk_total = usage.total
            disk_free = usage.free
        except Exception:
            disk_total = 0
            disk_free = 0

        max_disk_gb = int(self.settings.get("max_disk_gb", 500))

        return {
            "sizes": sizes,
            "total_used_bytes": total_used,
            "total_used_mb": round(total_used / (1024 * 1024), 1),
            "total_used_gb": round(total_used / (1024 * 1024 * 1024), 2),
            "disk_total_gb": round(disk_total / (1024 * 1024 * 1024), 1),
            "disk_free_gb": round(disk_free / (1024 * 1024 * 1024), 1),
            "max_disk_gb": max_disk_gb,
        }

    def force_set_password(self, new_password: str) -> bool:
        if not new_password or len(new_password) < 3:
            self.message.emit("New password is too short", "error")
            return False

        self.settings["password_hash"] = hash_password(new_password)
        self.db.save_setting("password_hash", self.settings["password_hash"])

        self.password_changed.emit()
        self.message.emit("Password changed", "success")

        log.info("Password force changed")

        return True