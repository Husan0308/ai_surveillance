from PySide6.QtCore import QObject, Signal

from backend.cameras.camera_worker import CameraWorker
from backend.cameras.connection_test import test_connection
from backend.core.logger import get_logger

log = get_logger("camera.manager")


class CameraManager(QObject):
    """
    Barcha kameralarni boshqaradi.

    - config/database dan kameralarni yuklaydi
    - har kamera uchun alohida worker thread yaratadi
    - kamera qo'shish / tahrirlash / o'chirish
    - connection test
    """

    status_changed = Signal(str, bool)
    frame_captured = Signal(str)
    health_updated = Signal(str, dict)

    camera_added = Signal(dict)
    camera_updated = Signal(dict)
    camera_removed = Signal(str)

    def __init__(self, config, db, event_bus=None):
        super().__init__()

        self.config = config
        self.db = db
        self.event_bus = event_bus

        self.cameras = {}
        self.workers = {}
        self.deepstream_enabled = bool(self.config.get("deepstream.enabled", False))
        self.deepstream_max_streams = int(
            self.config.get("deepstream.max_decode_streams", 3) or 3
        )
        if self.deepstream_enabled:
            log.info("DeepStream enabled: max %s NVDEC streams", self.deepstream_max_streams)

    # ---------------- load ----------------
    def load(self):
        # ✅ YAML USTUVOR: cameras.yaml dagi online qiymatini DB ga sinxronlash.
        # Endi kamerani YAML dan o'chirish/yoqish mumkin (dastur qayta ishga tushganda).
        try:
            import yaml as _yaml, os as _os
            _yp = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "config", "cameras.yaml")
            if _os.path.exists(_yp):
                with open(_yp, "r", encoding="utf-8") as _f:
                    _ycams = (_yaml.safe_load(_f) or {}).get("cameras", [])
                with self.db.lock:
                    for _yc in _ycams:
                        _cid = _yc.get("id")
                        if _cid and "online" in _yc:
                            self.db.conn.execute(
                                "UPDATE camera_config SET online=? WHERE id=?",
                                (1 if _yc.get("online") else 0, _cid))
                    self.db.conn.commit()
                print(f"[CameraManager] ✅ YAML→DB online sinxronlandi ({len(_ycams)} kamera)", flush=True)
        except Exception as _ye:
            print(f"[CameraManager] ⚠ YAML sinxron xato: {_ye}", flush=True)

        cams = self.db.get_camera_configs()

        if not cams:
            cams = self.config.load_cameras()
            for cam in cams:
                self.db.save_camera_config(cam)

        yaml_by_id = {
            item.get("id"): item for item in self.config.load_cameras()
            if item.get("id")
        }
        for cam in cams:
            yaml_cam = yaml_by_id.get(cam.get("id"), {})
            for key in ("source", "codec", "latency_ms", "resolution", "fps", "online"):
                if key in yaml_cam:
                    cam[key] = yaml_cam[key]
        for cam in cams:
            self.add_camera(cam, persist=False)

        log.info("CameraManager loaded %s cameras", len(self.cameras))

    # ---------------- add / update / delete ----------------
    def add_camera(self, cam: dict, persist: bool = True) -> bool:
        cid = cam.get("id")

        if not cid:
            log.error("Camera id is empty")
            return False

        if cid in self.workers:
            self.stop_camera(cid)

        self.cameras[cid] = cam

        if persist:
            self.db.save_camera_config(cam)

        active_deepstream = sum(
            1 for worker_id, existing in self.workers.items()
            if worker_id != cid and getattr(existing, "use_deepstream", False)
        )
        use_deepstream = (
            self.deepstream_enabled
            and str(cam.get("backend", "auto")).lower() != "opencv"
            and active_deepstream < self.deepstream_max_streams
        )
        # Parse per-camera target_size, default to 640x360 for AI-optimal bandwidth
        _res = cam.get("resolution", "640x360")
        if isinstance(_res, str) and "x" in _res:
            _rw, _rh = _res.lower().split("x")
            _target_size = (int(_rw), int(_rh))
        else:
            _target_size = (640, 360)
        worker = CameraWorker(
            cam,
            target_size=_target_size,
            use_deepstream=use_deepstream,
            use_gstreamer=not use_deepstream,
        )
        backend_name = "DeepStream/NVDEC" if use_deepstream else "GStreamer/software"
        log.info("Camera %s backend: %s", cid, backend_name)

        worker.status_changed.connect(self.status_changed)
        worker.frame_captured.connect(self.frame_captured)
        worker.health_updated.connect(self.health_updated)

        self.workers[cid] = worker

        if cam.get("online", False):
            worker.start()

        self.camera_added.emit(cam)
        log.info("Camera added: %s", cid)

        return True

    def update_camera(self, cam: dict) -> bool:
        cid = cam.get("id")

        if not cid:
            return False

        self.stop_camera(cid)
        self.add_camera(cam, persist=True)
        self.camera_updated.emit(cam)

        log.info("Camera updated: %s", cid)
        return True

    def delete_camera(self, camera_id: str):
        self.stop_camera(camera_id)

        self.workers.pop(camera_id, None)
        self.cameras.pop(camera_id, None)

        self.db.delete_camera_config(camera_id)
        self.camera_removed.emit(camera_id)

        log.info("Camera deleted: %s", camera_id)

    # ---------------- start / stop ----------------
    def start_camera(self, camera_id: str):
        worker = self.workers.get(camera_id)

        if worker is None:
            return

        worker.cfg["online"] = True
        self.cameras[camera_id]["online"] = True
        self.db.save_camera_config(worker.cfg)

        if not worker.isRunning():
            worker.start()

    def stop_camera(self, camera_id: str):
        worker = self.workers.get(camera_id)

        if worker is None:
            return

        worker.cfg["online"] = False

        if camera_id in self.cameras:
            self.cameras[camera_id]["online"] = False

        worker.stop()
        worker.wait(3000)

    # ---------------- test ----------------
    def test_connection(self, source, username=None, password=None, timeout: int = 5):
        return test_connection(source, username, password, timeout)

    # ---------------- access ----------------
    def get_worker(self, camera_id: str):
        return self.workers.get(camera_id)

    def get_camera(self, camera_id: str):
        return self.cameras.get(camera_id)

    def all_camera_ids(self):
        return list(self.cameras.keys())

    # ---------------- shutdown ----------------
    def shutdown(self):
        log.info("CameraManager shutting down")

        for worker in self.workers.values():
            worker.stop()

        for worker in self.workers.values():
            worker.wait(3000)