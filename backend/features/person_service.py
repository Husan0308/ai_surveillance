import random
import numpy as np

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont

from backend.core.logger import get_logger

log = get_logger("features.person")


def bytes_to_pixmap(data: bytes) -> QPixmap:
    pm = QPixmap()

    if data:
        pm.loadFromData(data)

    return pm


def pixmap_to_png_bytes(pm: QPixmap) -> bytes:
    if pm is None or pm.isNull():
        return b""

    from PySide6.QtCore import QByteArray, QBuffer

    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QBuffer.WriteOnly)
    pm.save(buf, "PNG")
    buf.close()

    return bytes(ba)


def jpg_bytes_from_bgr(bgr) -> bytes:
    if bgr is None:
        return b""

    import cv2

    ok, jpg = cv2.imencode(".jpg", bgr)

    if not ok:
        return b""

    return jpg.tobytes()


def make_default_avatar(name: str, size: int = 96) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(QColor(0, 0, 0, 0))

    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)

    colors = [
        "#2f7df6",
        "#8e44ad",
        "#16a085",
        "#d35400",
        "#c0392b",
        "#2980b9",
        "#27ae60",
    ]

    color = colors[(hash(name or "?") % len(colors))]

    p.setBrush(QColor(color))
    p.setPen(QColor(0, 0, 0, 0))
    p.drawEllipse(0, 0, size, size)

    initials = "".join([w[0] for w in str(name).split()[:2]]).upper() or "?"

    p.setPen(QColor("white"))
    p.setFont(QFont("Segoe UI", size // 3, QFont.Bold))
    p.drawText(pm.rect(), 0x0084, initials)  # AlignCenter

    p.end()

    return pm


class PersonService(QObject):
    """
    Person / Employee management service.

    - add
    - edit
    - delete
    - update faces
    - history
    - hot reload face gallery
    """

    persons_changed = Signal()
    person_added = Signal(dict)
    person_updated = Signal(dict)
    person_deleted = Signal(int)
    faces_updated = Signal(int)
    gallery_reloaded = Signal()

    def __init__(self, config, db, face_engine, db_writer=None, identity_manager=None):
        super().__init__()

        self.config = config
        self.db = db
        self.face_engine = face_engine
        self.db_writer = db_writer
        self.identity_manager = identity_manager

        self.delete_embeddings_on_delete = bool(
            config.get("person_management.delete_embeddings_on_delete", True)
        )

        log.info("PersonService started")

    # ---------------- helpers ----------------
    def _person_dict(self, row: dict) -> dict:
        d = dict(row)

        avatar_bytes = d.get("avatar")

        if avatar_bytes:
            pm = bytes_to_pixmap(avatar_bytes)
        else:
            pm = make_default_avatar(d.get("name", "?"))

        d["avatar_pm"] = pm

        return d

    def get_person_dict(self, person_id: int):
        from datetime import datetime, timedelta
        row = self.db.get_person(person_id)

        if row is None:
            return None

        d = self._person_dict(row)

        # Real-time online status
        now = datetime.now()
        ls = d.get("last_seen")
        live = False

        if self.identity_manager is not None:
            try:
                for cam_id, state in self.identity_manager.states.items():
                    for trk_id, trk_info in state.active_tracks.items():
                        pid = trk_info.get("person_id") if isinstance(trk_info, dict) else getattr(trk_info, "person_id", None)
                        if pid == person_id:
                            live = True
                            d["camera_id"] = cam_id
                            break
                    if live:
                        break
            except Exception:
                pass

        if live:
            d["online"] = True
            d["status_text"] = "Online"
        elif ls:
            try:
                ls_dt = datetime.fromisoformat(ls) if isinstance(ls, str) else ls
                elapsed = now - ls_dt
                if elapsed < timedelta(minutes=2):
                    d["online"] = True
                    d["status_text"] = "Online"
                else:
                    d["online"] = False
                    d["status_text"] = ls_dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                d["online"] = False
                d["status_text"] = "Unknown"
        else:
            d["online"] = False
            d["status_text"] = "Never seen"

        return d

    # ---------------- load ----------------
    def load_persons(self, identity_manager=None):
        """
        Personlarni yuklash + REAL-TIME status.
        identity_manager berilsa → live track ma'lumotidan online aniqlanadi.
        """
        from datetime import datetime, timedelta
        rows = self.db.get_persons()
        result = [self._person_dict(r) for r in rows]

        now = datetime.now()
        online_threshold = timedelta(minutes=2)

        # Live: qaysi person_id lar hozir kamerada?
        live_person_ids = set()
        live_camera_map = {}  # person_id → camera_id

        im = identity_manager or self.identity_manager
        if im is not None:
            try:
                for cam_id, state in im.states.items():
                    for trk_id, trk_info in state.active_tracks.items():
                        pid = trk_info.get("person_id") if isinstance(trk_info, dict) else getattr(trk_info, "person_id", None)
                        if pid is not None:
                            live_person_ids.add(pid)
                            live_camera_map[pid] = cam_id
            except Exception:
                pass

        for p in result:
            pid = p.get("id")
            ls = p.get("last_seen")

            # ✅ REAL-TIME: IdentityManager dan live status
            if pid in live_person_ids:
                p["online"] = True
                p["camera_id"] = live_camera_map.get(pid)
                p["status_text"] = "Online"
                p["last_seen_dt"] = now
            elif ls is None or ls == "":
                p["online"] = False
                p["camera_id"] = None
                p["last_seen_dt"] = None
                p["status_text"] = "Never seen"
            else:
                try:
                    if isinstance(ls, str):
                        ls_dt = datetime.fromisoformat(ls)
                    else:
                        ls_dt = ls
                    p["last_seen_dt"] = ls_dt
                    elapsed = now - ls_dt

                    if elapsed < online_threshold:
                        p["online"] = True
                        p["status_text"] = "Online"
                    else:
                        p["online"] = False
                        if elapsed < timedelta(hours=1):
                            p["status_text"] = f"{int(elapsed.total_seconds()//60)} min ago"
                        elif elapsed < timedelta(hours=24):
                            p["status_text"] = f"{int(elapsed.total_seconds()//3600)}h ago"
                        else:
                            p["status_text"] = ls_dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    p["online"] = False
                    p["last_seen_dt"] = None
                    p["status_text"] = "Unknown"

            if "camera_id" not in p or p.get("camera_id") is None:
                p["camera_id"] = None
            p.setdefault("track_id", None)

        return result

    # ---------------- add ----------------

    def get_all_persons(self):
        """Barcha person larni DB dan yuklash (auto sync uchun)"""
        try:
            rows = self.db.conn.execute("SELECT * FROM persons ORDER BY name").fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[PersonService] ⚠ get_all_persons error: {e}", flush=True)
            return []

    def add_person(
        self,
        name: str,
        department: str,
        employee_id: str = "",
        avatar_bytes: bytes = None,
        images_bgr: list = None,
    ):
        name = (name or "").strip()

        if not name:
            return None

        if not employee_id:
            employee_id = f"EMP-{random.randint(1000, 9999)}"

        if not avatar_bytes and images_bgr:
            avatar_bytes = jpg_bytes_from_bgr(images_bgr[0])

        person = {
            "name": name,
            "department": department,
            "employee_id": employee_id,
            "status": "Active",
            "avatar": avatar_bytes,
            "last_seen": None,
            "rec_count": 0,
            "stay_total": 0,
        }

        try:
            person_id = self.db.save_person(person)
        except Exception as e:
            log.error("add_person error: %s", e)
            return None

        if images_bgr:
            self.update_faces(person_id, images_bgr)

        data = self.get_person_dict(person_id)

        self.person_added.emit(data)
        self.persons_changed.emit()

        log.info("Person added: %s (%s)", name, employee_id)

        return person_id

    # ---------------- update ----------------
    def update_person(
        self,
        person_id: int,
        name: str,
        department: str,
        employee_id: str,
        status: str = "Active",
        avatar_bytes: bytes = None,
    ):
        existing = self.db.get_person(person_id)

        if existing is None:
            return False

        person = {
            "id": person_id,
            "name": name or existing.get("name"),
            "department": department if department is not None else existing.get("department"),
            "employee_id": employee_id or existing.get("employee_id"),
            "status": status or existing.get("status", "Active"),
            "avatar": avatar_bytes if avatar_bytes is not None else existing.get("avatar"),
            "last_seen": existing.get("last_seen"),
            "rec_count": existing.get("rec_count", 0),
            "stay_total": existing.get("stay_total", 0),
        }

        try:
            self.db.save_person(person)
        except Exception as e:
            log.error("update_person error: %s", e)
            return False

        # update gallery names
        self._reload_person_gallery(person_id)

        data = self.get_person_dict(person_id)

        self.person_updated.emit(data)
        self.persons_changed.emit()

        log.info("Person updated: %s", person_id)

        return True

    # ---------------- delete ----------------
    def delete_person(self, person_id: int):
        try:
            if self.delete_embeddings_on_delete:
                # foreign key cascade should delete embeddings and visits
                self.db.delete_person(person_id)
            else:
                self.db.delete_embeddings(person_id)
                self.db.delete_person(person_id)

        except Exception as e:
            log.error("delete_person error: %s", e)
            return False

        # remove from live recognition gallery
        try:
            self.face_engine.remove_person(person_id)
        except Exception as e:
            log.error("delete_person gallery error: %s", e)

        self.person_deleted.emit(person_id)
        self.persons_changed.emit()

        log.info("Person deleted: %s", person_id)

        return True

    # ---------------- faces ----------------
    def update_faces(self, person_id: int, images_bgr: list):
        """
        Replace all face embeddings for person.
        """

        if not images_bgr:
            return False

        person = self.db.get_person(person_id)

        if person is None:
            return False

        embeddings = []
        images = []
        scores = []

        for img in images_bgr:
            if img is None:
                continue

            faces = self.face_engine.detect(img, need_embedding=True)

            if not faces:
                continue

            face = max(
                faces,
                key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
            )

            q = self.face_engine.quality(img, face)

            if q.get("score", 0) < 40:
                continue

            if face["embedding"] is not None:
                embeddings.append(face["embedding"])
                images.append(img)
                scores.append(q.get("score", 0))

        if not embeddings:
            return False

        try:
            # delete old embeddings
            self.db.delete_embeddings(person_id)

            # remove old gallery entries
            self.face_engine.remove_person(person_id)

            # add new embeddings
            for emb, img, score in zip(embeddings, images, scores):
                emb_bytes = np.asarray(emb, dtype=np.float32).tobytes()
                img_bytes = jpg_bytes_from_bgr(img)

                self.db.add_embedding(
                    person_id=person_id,
                    embedding=emb_bytes,
                    image=img_bytes,
                    quality=float(score),
                )

                self.face_engine.add_to_gallery(
                    person_id=person_id,
                    name=person.get("name", ""),
                    embedding=emb,
                )

        except Exception as e:
            log.error("update_faces error: %s", e)
            return False

        self.faces_updated.emit(person_id)
        self.gallery_reloaded.emit()

        log.info("Person faces updated: %s (%s embeddings)", person_id, len(embeddings))

        return True

    def add_face_images(self, person_id: int, images_bgr: list):
        """
        Add additional face embeddings without deleting old ones.
        """

        if not images_bgr:
            return False

        person = self.db.get_person(person_id)

        if person is None:
            return False

        added = 0

        for img in images_bgr:
            if img is None:
                continue

            faces = self.face_engine.detect(img, need_embedding=True)

            if not faces:
                continue

            face = max(
                faces,
                key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
            )

            q = self.face_engine.quality(img, face)

            if q.get("score", 0) < 40:
                continue

            if face["embedding"] is None:
                continue

            try:
                emb_bytes = np.asarray(face["embedding"], dtype=np.float32).tobytes()
                img_bytes = jpg_bytes_from_bgr(img)

                self.db.add_embedding(
                    person_id=person_id,
                    embedding=emb_bytes,
                    image=img_bytes,
                    quality=float(q.get("score", 0)),
                )

                self.face_engine.add_to_gallery(
                    person_id=person_id,
                    name=person.get("name", ""),
                    embedding=face["embedding"],
                )

                added += 1

            except Exception as e:
                log.error("add_face_images error: %s", e)

        if added > 0:
            self.faces_updated.emit(person_id)
            self.gallery_reloaded.emit()

        return added > 0

    # ---------------- gallery ----------------
    def _reload_person_gallery(self, person_id: int):
        try:
            person = self.db.get_person(person_id)

            if person is None:
                return

            self.face_engine.remove_person(person_id)

            rows = self.db.get_embeddings_by_person(person_id)

            for row in rows:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)

                self.face_engine.add_to_gallery(
                    person_id=person_id,
                    name=person.get("name", ""),
                    embedding=emb,
                )

            self.gallery_reloaded.emit()

        except Exception as e:
            log.error("_reload_person_gallery error: %s", e)

    def reload_full_gallery(self):
        try:
            self.face_engine.load_gallery()
            self.gallery_reloaded.emit()
        except Exception as e:
            log.error("reload_full_gallery error: %s", e)

    # ---------------- history ----------------
    def get_recognition_history(self, person_id: int, limit: int = 200):
        return self.db.get_recognition_history(person_id, limit)

    def get_visit_history(self, person_id: int, limit: int = 200):
        return self.db.get_visits(person_id, limit)

    def get_events_history(self, person_id: int, limit: int = 200):
        return self.db.get_events_by_person(person_id, limit)

    def get_timeline(self, person_id: int):
        return self.db.get_person_timeline(person_id)

    
    def _get_realtime_timeline(self, person_id: int):
        """Bugungi kun uchun soatbay (0-23) real-time ma'lumot"""
        from datetime import datetime
        timeline = [0] * 24
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            with self.db.lock:
                # events jadvalidan bugungi kun uchun soatbay hisoblash
                rows = self.db.conn.execute("""
                    SELECT CAST(strftime('%H', time) AS INTEGER) as hour, COUNT(DISTINCT strftime('%M', time)) as count
                    FROM events
                    WHERE person_id = ? AND DATE(time) = ?
                    GROUP BY hour
                """, (person_id, today)).fetchall()
                
                for row in rows:
                    hour = row["hour"]
                    count = row["count"]
                    if hour is not None and 0 <= hour < 24:
                        timeline[hour] = count
                
                # Agar events bo'sh bo'lsa, face_events dan sinab ko'rish
                if sum(timeline) == 0:
                    rows2 = self.db.conn.execute("""
                        SELECT CAST(strftime('%H', created_at) AS INTEGER) as hour, COUNT(*) as count
                        FROM face_embeddings
                        WHERE person_id = ? AND DATE(created_at) = ?
                        GROUP BY hour
                    """, (person_id, today)).fetchall()
                    for row in rows2:
                        hour = row["hour"]
                        count = row["count"]
                        if hour is not None and 0 <= hour < 24:
                            timeline[hour] = count
        except Exception as e:
            print(f"[PersonService] timeline error: {e}", flush=True)
        return timeline

def get_full_profile(self, person_id: int):
        person = self.get_person_dict(person_id)

        if person is None:
            return None

        recognition = self.get_recognition_history(person_id, 100)
        visits = self.get_visit_history(person_id, 100)
        timeline = self.get_timeline(person_id)
        events = self.get_events_history(person_id, 100)

        total_stay = sum(int(v.get("duration_sec", 0) or 0) for v in visits)

        return {
            "person": person,
            "recognition_history": recognition,
            "visit_history": visits,
            "timeline": timeline,
            "events": events,
            "total_stay_sec": total_stay,
        }