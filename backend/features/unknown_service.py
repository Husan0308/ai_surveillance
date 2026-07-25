import numpy as np

from PySide6.QtCore import QObject, Signal

from backend.features.person_service import bytes_to_pixmap
from backend.core.logger import get_logger

log = get_logger("features.unknown")


class UnknownService(QObject):
    """
    Unknown faces service.

    - unknown faces list
    - convert unknown -> employee
    - delete unknown face
    """

    unknown_changed = Signal()
    unknown_converted = Signal(int, int)   # unknown_id, person_id
    unknown_deleted = Signal(int)

    def __init__(self, config, db, face_engine, person_service):
        super().__init__()

        self.config = config
        self.db = db
        self.face_engine = face_engine
        person_service_ref = person_service
        self.person_service = person_service_ref

        log.info("UnknownService started")

    # ---------------- list ----------------
    def get_unknown_faces(self, limit: int = 200):
        rows = self.db.get_unknown_faces(limit)

        out = []

        for r in rows:
            d = dict(r)

            d["image_pm"] = bytes_to_pixmap(d.get("image"))

            out.append(d)

        return out

    # ---------------- convert ----------------
    def convert_unknown_to_person(
        self,
        unknown_id: int,
        name: str,
        department: str,
        employee_id: str = "",
    ):
        row = self.db.get_unknown_face(unknown_id)

        if row is None:
            return None

        name = (name or "").strip()

        if not name:
            return None

        image_bytes = row.get("image")
        embedding_bytes = row.get("embedding")

        # create person using unknown image as avatar
        person_id = self.person_service.add_person(
            name=name,
            department=department,
            employee_id=employee_id,
            avatar_bytes=image_bytes,
            images_bgr=None,
        )

        if person_id is None:
            return None

        # attach embedding if exists
        if embedding_bytes:
            try:
                self.db.add_embedding(
                    person_id=person_id,
                    embedding=embedding_bytes,
                    image=image_bytes,
                    quality=0.0,
                )

                emb = np.frombuffer(embedding_bytes, dtype=np.float32)

                self.face_engine.add_to_gallery(
                    person_id=person_id,
                    name=name,
                    embedding=emb,
                )

            except Exception as e:
                log.error("convert_unknown_to_person embedding error: %s", e)

        # mark unknown as converted
        try:
            self.db.convert_unknown_to_person(unknown_id, person_id)
        except Exception as e:
            log.error("convert_unknown_to_person mark error: %s", e)

        self.unknown_converted.emit(unknown_id, person_id)
        self.unknown_changed.emit()

        log.info("Unknown %s converted to person %s", unknown_id, person_id)

        return person_id

    # ---------------- delete ----------------
    def delete_unknown_face(self, unknown_id: int):
        try:
            self.db.delete_unknown_face(unknown_id)
        except Exception as e:
            log.error("delete_unknown_face error: %s", e)
            return False

        self.unknown_deleted.emit(unknown_id)
        self.unknown_changed.emit()

        return True