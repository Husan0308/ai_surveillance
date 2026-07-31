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

        self.enabled = bool(config.get("enabled", config.get("ai.face.enabled", True)))
        self.threshold = float(config.get("threshold", config.get("match_threshold", config.get("ai.face.match_threshold", 0.45))))
        self.min_face_size = int(config.get("min_face_size", config.get("min_face_size_px", config.get("ai.face.min_face_size_px", 8))))
        _ds = config.get("det_size", config.get("ai.face.det_size", 1280))
        if isinstance(_ds, (list, tuple)):
            self.det_size = int(_ds[0])
        else:
            self.det_size = int(_ds)

        self.app = None
        self.available = False
        self.lock = threading.Lock()

        self.gallery = []

        if not self.enabled:
            log.info("Face engine disabled")
            return

        try:
            from insightface.app import FaceAnalysis

            model_name = config.get("model", config.get("ai.face.model", "buffalo_l"))

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
                det_size=(
                self.det_size if isinstance(self.det_size, int) else self.det_size[0],
                self.det_size if isinstance(self.det_size, int) else self.det_size[1],
            ),
            )

            self.available = True
            log.info("Face engine loaded: %s", model_name)

            self.load_gallery()

        except Exception as e:
            log.error("Face engine unavailable: %s", e)

    # ---------------- gallery ----------------
    def load_gallery(self):
        """DB dan face embedding larni yuklash (tuple va dict format qo'llab-quvvatlaydi)"""
        self.gallery = []
        if self.db is None:
            log.warning("load_gallery: db is None!")
            return
        try:
            # DB wrapper yoki raw connection
            if hasattr(self.db, 'get_embeddings'):
                rows = self.db.get_embeddings()
            else:
                cur = self.db.cursor()
                cur.execute("""
                    SELECT fe.person_id, fe.embedding, p.name
                    FROM face_embeddings fe
                    LEFT JOIN persons p ON p.id = fe.person_id
                    WHERE fe.embedding IS NOT NULL AND LENGTH(fe.embedding) > 100
                    ORDER BY COALESCE(fe.quality, 0) DESC
                """)
                rows = cur.fetchall()

            seen_ids = set()
            for row in rows:
                # Tuple yoki dict formatni qo'llab-quvvatlash
                if isinstance(row, dict):
                    pid = row.get("person_id")
                    raw_emb = row.get("embedding")
                    pname = row.get("name", "")
                else:
                    # Raw SQL tuple: (person_id, embedding, name)
                    pid = row[0]
                    raw_emb = row[1]
                    pname = row[2] if len(row) > 2 else ""

                if pid in seen_ids or raw_emb is None:
                    continue

                import numpy as np
                emb = np.frombuffer(raw_emb, dtype=np.float32)
                if len(emb) >= 128:
                    self.gallery.append((pid, pname or f"Person-{pid}", emb))
                    seen_ids.add(pid)

            log.info("Face gallery loaded: %d embeddings", len(self.gallery))
        except Exception as e:
            log.error("load_gallery error: %s", e)

    def match(self, embedding, threshold=None):
        """Embedding ni gallery bilan solishtirish"""
        if embedding is None or not self.gallery:
            return None
        
        thresh = threshold or self.threshold
        best = None
        best_sim = 0.0
        
        for entry in self.gallery:
            # Tuple (pid, name, emb) yoki dict format
            if isinstance(entry, dict):
                gal_emb = entry.get("embedding")
            else:
                gal_emb = entry[2]  # tuple

            if gal_emb is None:
                continue
            
            dot = float(np.dot(embedding, gal_emb))
            norm_e = float(np.linalg.norm(embedding))
            norm_g = float(np.linalg.norm(gal_emb))
            sim = dot / (norm_e * norm_g + 1e-8)
            
            if sim > best_sim:
                best_sim = sim
                best = entry
        
        if best and best_sim >= thresh:
            if isinstance(best, dict):
                return {
                    "person_id": best.get("person_id"),
                    "name": best.get("name", ""),
                    "similarity": best_sim,
                }
            else:
                return {
                    "person_id": best[0],
                    "name": best[1],
                    "similarity": best_sim,
                }
        return None

    def add_to_gallery(self, person_id, name, embedding):
        """Gallery ga qo'shish + DB ga saqlash (persistent)"""
        emb = self._normalize(embedding)
        if emb is None:
            return

        # Memory: duplicate tekshirish
        for i, g in enumerate(self.gallery):
            gid = g.get("person_id") if isinstance(g, dict) else g[0]
            if gid == person_id:
                self.gallery[i] = {"person_id": person_id, "name": name, "embedding": emb}
                return

        self.gallery.append({"person_id": person_id, "name": name, "embedding": emb})
        log.info("Gallery +: %s (id=%s) total=%d", name, person_id, len(self.gallery))

        # DB ga saqlash (qayta run ham tanishi uchun)
        if self.db is not None:
            try:
                import numpy as _np
                emb_bytes = emb.astype(_np.float32).tobytes()
                if hasattr(self.db, 'conn'):
                    conn = self.db.conn
                elif hasattr(self.db, 'get_connection'):
                    conn = self.db.get_connection()
                else:
                    conn = self.db
                cur = conn.cursor()
                # Eski embedding ni o'chirish
                cur.execute("DELETE FROM face_embeddings WHERE person_id = ?", (person_id,))
                # Yangi qo'shish
                cur.execute(
                    "INSERT INTO face_embeddings (person_id, embedding, quality) VALUES (?, ?, ?)",
                    (person_id, emb_bytes, 80.0)
                )
                conn.commit()
                log.info("DB saved embedding: %s (id=%s)", name, person_id)
            except Exception as e:
                log.warning("DB save embedding error: %s", e)


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
        """Face detection — barcha filtrlar olib tashlandi (debug uchun)"""
        if not self.enabled or not self.available or self.app is None or bgr is None:
            return []

        # Frame ni MAJBURIY contiguous uint8 qilish
        import numpy as _np
        if not bgr.flags['C_CONTIGUOUS']:
            bgr = _np.ascontiguousarray(bgr)
        if bgr.dtype != _np.uint8:
            bgr = bgr.astype(_np.uint8)

        with self.lock:
            try:
                faces = self.app.get(bgr)
            except Exception as e:
                log.error("Face detection error: %s", e)
                return []

        raw_count = len(faces)
        out = []

        for f in faces:
            try:
                bbox = [float(x) for x in getattr(f, "bbox", [])]
                if len(bbox) != 4:
                    continue

                emb = getattr(f, "normed_embedding", None)
                if emb is None:
                    emb = getattr(f, "embedding", None)
                emb = self._normalize(emb)

                if need_embedding and emb is None:
                    continue

                out.append({
                    "bbox": bbox,
                    "det_score": float(getattr(f, "det_score", 0.0) or 0.0),
                    "embedding": emb,
                    "landmarks": getattr(f, "kps", None),
                    "pose": getattr(f, "pose", None),
                })
            except Exception:
                continue

        # DEBUG: har safar log
        if raw_count > 0 or True:
            log.info("Face detect: raw=%d passed=%d frame_shape=%s", raw_count, len(out), getattr(bgr, 'shape', '?'))

        return out

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
            # Tuple (pid, name, emb) yoki dict format
            if isinstance(g, dict):
                gal_emb = g.get("embedding")
            else:
                gal_emb = g[2]  # tuple: (pid, name, embedding)

            if gal_emb is None:
                continue

            score = float(np.dot(emb, gal_emb))
            scores.append((score, g))

        scores.sort(key=lambda x: x[0], reverse=True)

        # Agar biror biri threshold dan yuqori bo'lsa — tanildi
        for score, g in scores[:5]:
            if score >= self.threshold:
                if isinstance(g, dict):
                    return g.get("person_id"), g.get("name", ""), score
                else:
                    return g[0], g[1], score  # tuple: (pid, name, emb)

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