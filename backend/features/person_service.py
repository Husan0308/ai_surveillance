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

    def __init__(self, config, db, face_engine, db_writer=None):
        super().__init__()

        self.config = config
        self.db = db
        self.face_engine = face_engine
        self.db_writer = db_writer

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
        row = self.db.get_person(person_id)

        if row is None:
            return None

        return self._person_dict(row)

    # ---------------- load ----------------
    def load_persons(self):
        print(f"[PersonService] load_persons called", flush=True)
        rows = self.db.get_persons()
        print(f"[PersonService] DB returned {len(rows)} persons", flush=True)
        
        result = [self._person_dict(r) for r in rows]
        # Default: barcha persons OFFLINE
        for p in result:
            p["online"] = False
            p["camera_id"] = None
            p["track_id"] = None
            p["last_seen_dt"] = p.get("last_seen")
        print(f"[PersonService] Returning {len(result)} person dicts (all offline)", flush=True)
        
        return result

    # ---------------- add ----------------
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