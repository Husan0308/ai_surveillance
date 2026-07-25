import threading
import numpy as np
import cv2

from backend.core.logger import get_logger

log = get_logger("ai.face")


class FaceEngine:
    """
    InsightFace engine.

    Face detection:
    - bbox
    - embedding
    - quality: blur, brightness, angle, size

    Recognition:
    - gallery from database
    - hot reload
    """

    def __init__(self, config, db=None):
        self.config = config
        self.db = db

        self.enabled = bool(config.get("ai.face.enabled", True))
        self.threshold = float(config.get("ai.face.match_threshold", 0.58))
        self.min_face_size = int(config.get("ai.face.min_face_size_px", 60))
        self.det_size = int(config.get("ai.face.det_size", 320))

        self.app = None
        self.available = False
        self.lock = threading.Lock()

        self.gallery = []

        if not self.enabled:
            log.info("Face engine disabled")
            return

        try:
            from insightface.app import FaceAnalysis

            model_name = config.get("ai.face.model", "buffalo_l")

            self.app = FaceAnalysis(
                name=model_name,
                # Auto GPU/CPU tanlash
                providers=(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if "CUDAExecutionProvider" in __import__("onnxruntime").get_available_providers()
                    else ["CPUExecutionProvider"]
                ),
            )

            self.app.prepare(
                ctx_id=0 if "CUDAExecutionProvider" in __import__("onnxruntime").get_available_providers() else -1,
                det_size=(self.det_size, self.det_size),
            )

            self.available = True
            log.info("Face engine loaded: %s", model_name)

            self.load_gallery()

        except Exception as e:
            log.error("Face engine unavailable: %s", e)

    # ---------------- gallery ----------------
    def load_gallery(self):
        if self.db is None:
            return

        try:
            rows = self.db.get_embeddings()
            self.gallery = []

            for row in rows:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                emb = self._normalize(emb)

                self.gallery.append({
                    "person_id": row["person_id"],
                    "name": row["name"],
                    "embedding": emb,
                })

            log.info("Face gallery loaded: %s embeddings", len(self.gallery))

        except Exception as e:
            log.error("load_gallery error: %s", e)

    def add_to_gallery(self, person_id, name, embedding):
        emb = self._normalize(embedding)

        if emb is None:
            return

        self.gallery.append({
            "person_id": person_id,
            "name": name,
            "embedding": emb,
        })

    def remove_person(self, person_id):
        self.gallery = [g for g in self.gallery if g["person_id"] != person_id]

    # ---------------- utils ----------------
    def _normalize(self, emb):
        if emb is None:
            return None

        emb = np.asarray(emb, dtype=np.float32)
        n = np.linalg.norm(emb)

        if n == 0:
            return None

        return emb / n

    # ---------------- detection ----------------
    def detect(self, bgr, need_embedding=True):
        if not self.enabled or not self.available or self.app is None or bgr is None:
            return []

        with self.lock:
            try:
                faces = self.app.get(bgr)
            except Exception as e:
                log.error("Face detection error: %s", e)
                return []

        out = []

        for f in faces:
            try:
                bbox = [float(x) for x in getattr(f, "bbox", [])]

                if len(bbox) != 4:
                    continue

                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]

                if w < self.min_face_size or h < self.min_face_size:
                    continue

                emb = getattr(f, "normed_embedding", None)

                if emb is None:
                    emb = getattr(f, "embedding", None)

                emb = self._normalize(emb)

                if need_embedding and emb is None:
                    continue

                out.append({
                    "bbox": bbox,
                    "score": float(getattr(f, "det_score", 0.0)),
                    "embedding": emb,
                    "landmarks": getattr(f, "kps", None),
                    "pose": getattr(f, "pose", None),
                })

            except Exception:
                continue

        return out

    # ---------------- quality ----------------
    def quality(self, bgr, face):
        result = {
            "score": 0.0,
            "blur_ok": False,
            "brightness_ok": False,
            "angle_ok": False,
            "size_ok": False,
        }

        try:
            x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
            h, w = bgr.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return result

            crop = bgr[y1:y2, x1:x2]

            if crop.size == 0:
                return result

            # size
            fw = x2 - x1
            fh = y2 - y1
            result["size_ok"] = fw >= self.min_face_size and fh >= self.min_face_size

            # blur
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            result["blur_ok"] = blur_score > 80.0

            # brightness
            brightness = float(gray.mean())
            result["brightness_ok"] = 45.0 < brightness < 210.0

            # angle
            pose = face.get("pose")
            if pose is not None:
                pitch, yaw, roll = [abs(float(v)) for v in pose]
                result["angle_ok"] = yaw < 30.0 and pitch < 25.0 and roll < 30.0
            else:
                result["angle_ok"] = True

            score = 0.0
            score += 30.0 if result["size_ok"] else 0.0
            score += 30.0 if result["blur_ok"] else 0.0
            score += 20.0 if result["brightness_ok"] else 0.0
            score += 20.0 if result["angle_ok"] else 0.0

            result["score"] = score

        except Exception as e:
            log.error("face quality error: %s", e)

        return result

    # ---------------- recognition ----------------
    def recognize(self, embedding):
        emb = self._normalize(embedding)

        if emb is None or not self.gallery:
            return None, "Unknown", 0.0

        # Top-5 eng yaqin embeddinglarni topish
        scores = []
        for g in self.gallery:
            score = float(np.dot(emb, g["embedding"]))
            scores.append((score, g))

        scores.sort(key=lambda x: x[0], reverse=True)

        # Agar biror biri threshold dan yuqori bo'lsa — tanildi
        for score, g in scores[:5]:
            if score >= self.threshold:
                return g["person_id"], g["name"], score

        # Hech biri threshold dan o'tmasa — eng yaxshisini qaytarish
        if scores:
            best_score, best_g = scores[0]
            return None, "Unknown", max(0.0, best_score)

        return None, "Unknown", 0.0
    
    # ---------------- registration ----------------
    def compute_embedding_from_images(self, images_bgr):
        embeddings = []
        scores = []

        for img in images_bgr:
            if img is None:
                continue

            faces = self.detect(img, need_embedding=True)

            if not faces:
                continue

            face = max(
                faces,
                key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
            )

            q = self.quality(img, face)

            if q["score"] < 50:
                continue

            if face["embedding"] is not None:
                embeddings.append(face["embedding"])
                scores.append(q["score"])

        if not embeddings:
            return None, 0.0

        avg = np.mean(embeddings, axis=0).astype(np.float32)
        avg = self._normalize(avg)

        return avg, float(np.mean(scores))