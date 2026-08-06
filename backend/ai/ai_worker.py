import time
import threading

from dataclasses import dataclass, field

from PySide6.QtCore import QThread, Signal

from backend.ai.tracker import iou
from backend.core.logger import get_logger

log = get_logger("ai.worker")


# ============================ DATA ================================
@dataclass
class DetectedPerson:
    track_id: int
    box: list
    conf: float
    known: bool = False
    person_id: int = None
    name: str = "Unknown"
    face_conf: float = 0.0
    reid_score: float = 0.0
    ankle: tuple = None
    age: float = 0.0
    first_seen: float = 0.0


@dataclass
class AIResult:
    camera_id: str
    frame_id: int
    timestamp: float
    persons: list = field(default_factory=list)
    infer_ms: float = 0.0
    frame: object = None


# ============================ IDENTITY CACHE ================================
class IdentityCache:
    """
    Har Track ID uchun identity saqlaydi.

    Track ID active ekan, cached identity qayta ishlatiladi.
    Face recognition har frame chaqirilmaydi.
    """

    def __init__(self):
        self.cache = {}

    def get(self, track_id):
        return self.cache.get(track_id)

    def set(
        self,
        track_id,
        person_id,
        name,
        embedding,
        confidence,
        camera_id,
        frame_id,
    ):
        self.cache[track_id] = {
            "person_id": person_id,
            "name": name,
            "embedding": embedding,
            "confidence": confidence,
            "last_recognition_time": time.time(),
            "last_seen_frame": frame_id,
            "camera_id": camera_id,
            "source": "face",
        }

    def update_seen(self, track_id, frame_id):
        c = self.cache.get(track_id)

        if c is not None:
            c["last_seen_frame"] = frame_id

    def remove(self, track_id):
        self.cache.pop(track_id, None)

    def active_ids(self):
        return set(self.cache.keys())


# ============================ AI WORKER ================================
class AIWorker(QThread):
    """
    Multi-stage AI pipeline.

    Stage 1: YOLO detection har N frame
    Stage 2: ByteTrack tracking
    Stage 3: Face Recognition faqat kerak bo‘lganda
    Stage 4: ReID (optional)
    Stage 5: Identity Cache
    Stage 6: Track termination
    """

    result_ready = Signal(str, object)
    event_detected = Signal(str, dict)

    def __init__(
        self,
        camera_id,
        frame_buffer,
        detector,
        tracker,
        pose_engine,
        face_engine,
        reid_engine,
        config,
        db_writer=None,
    ):
        super().__init__()

        self.camera_id = camera_id
        self.buffer = frame_buffer

        self.detector = detector
        self.tracker = tracker
        self.pose = pose_engine
        self.face = face_engine
        self.reid = reid_engine

        # ===== ReID + Tracking: body feature gallery =====
        self._emitted = set()           # (event_type, camera_id, track_id)
        self._unknown_zone_ts = []
        self._recent_person_event_boxes = []
        self.shared_gallery = None      # 🌐 cross-camera gallery (ServiceManager set qiladi)
        self.unknown_registry = None    # shared face-embedding unknown IDs
        self.global_identity_cache = None
        self.body_gallery = {}          # person_id → [HSV histogram feature]
        # ✅ SHARED unknown gallery (barcha kamera bitta — cross-camera unknown ReID)
        if not hasattr(type(self), "_shared_unknown_gallery"):
            type(self)._shared_unknown_gallery = {}
            type(self)._shared_unknown_lock = threading.RLock()
            type(self)._shared_unknown_next_id = 1
            type(self)._shared_unknown_reservations = {}
        self.unknown_gallery = type(self)._shared_unknown_gallery
        self._current_track_ids = set()   # hozirgi frame dagi barcha track id
        self._recent_recognized = []   # 🧠 [(box4,person_id,name,time)] yuz burilgan odam xotirasi
        self.body_gallery_names = {}    # person_id → name
        self.track_body_features = {}
        self._reid_last_ts = {}   # track_id → oxirgi body feature
        try:
            self.reid_threshold = float(config.get("ai", {}).get("reid", {}).get("threshold", 0.65))
        except Exception:
            self.reid_threshold = 0.65
        # 🗺️ Xona mapping (co-occurrence mantiq uchun)
        try:
            self.camera_rooms = config.get("camera_rooms", {}) if hasattr(config, "get") else {}
        except Exception:
            self.camera_rooms = {}
        if not self.camera_rooms and isinstance(config, dict):
            self.camera_rooms = config.get("camera_rooms", {})
        self.room = self.camera_rooms.get(self.camera_id, self.camera_id)
        try:
            self.min_transition = float(config.get("ai", {}).get("reid", {}).get("min_transition_seconds", 20))
        except Exception:
            self.min_transition = 20.0
        print(f"[AIWorker {self.camera_id}] 🗺 room={self.room} min_transition={self.min_transition}s", flush=True)

        self.config = config
        self.db_writer = db_writer

        self.detection_interval = int(config.get("ai", {}).get("detection_interval_frames", 3))
        self.face_timeout = float(config.get("ai", {}).get("face_recognition_timeout_sec", 7.0))
        self.face_min_quality = float(config.get("ai", {}).get("face_min_quality", 50.0))
        self.reid_timeout = float(config.get("ai", {}).get("reid_timeout_sec", 15.0))

        self.identity_cache = IdentityCache()

        self.frame_count = 0
        self.last_persons = []

        self._running = False

        # ===== MEGA BATCH =====
        self.batch_scheduler = None
        self._detection_result = None
        self._detection_lock = threading.Lock()
        self._detection_event = threading.Event()

    def stop(self):
        self._running = False


    def set_batch_scheduler(self, scheduler):
        """Batch scheduler ni sozlash."""
        self.batch_scheduler = scheduler

    def _on_batch_detections(self, detections):
        """Batch scheduler dan natijani qabul qilish."""
        with self._detection_lock:
            self._detection_result = detections
            self._detection_event.set()


    # ---------------- main loop ----------------
    def run(self):
        self._running = True
        ai_fps = int(self.config.get("ai", {}).get("ai_fps", 8))
        interval = 1.0 / max(1, ai_fps)

        pose_ok = self.pose.available if self.pose else False
        det_ok = self.detector.available if self.detector else False
        print(f"[AIWorker {self.camera_id}] THREAD STARTED | Pose={pose_ok} | Det={det_ok} | ai_fps={ai_fps} | det_interval={self.detection_interval}", flush=True)

        last_fid = -1

        while self._running:
            loop_t = time.time()
            frame, ts, fid = self.buffer.get()

            if frame is None:
                time.sleep(0.05)
                continue

            # Yangi frame kelganmi? (bir xil frame ni qayta ishlamaslik)
            if fid == last_fid:
                time.sleep(0.01)
                continue
            last_fid = fid

            self.frame_count += 1

            run_detection = (self.frame_count % self.detection_interval == 0)

            try:
                persons = self._run_pipeline(frame, fid, do_detection=run_detection)
                self.last_persons = persons
            except Exception as _pe:
                # ✅ GLOBAL FIX: thread hech qachon o'lmaydi
                print(f"[AIWorker {self.camera_id}] ⚠ pipeline xato (thread yashaydi): {_pe}", flush=True)
                import traceback as _tb; _tb.print_exc()
                persons = self.last_persons

            infer_ms = (time.time() - loop_t) * 1000.0

            result = AIResult(
                camera_id=self.camera_id,
                frame_id=fid,
                timestamp=time.time(),
                persons=persons,
                infer_ms=infer_ms,
                frame=frame,
            )

            self.result_ready.emit(self.camera_id, result)

            elapsed = time.time() - loop_t
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print(f"[AIWorker {self.camera_id}] THREAD STOPPED", flush=True)

    # ---------------- pipeline ----------------
    def _run_pipeline(self, frame, fid, do_detection=True):
        # ---------------- Stage 1: Clean & Direct Person Detection ----------------
        kpts = []
        boxes = []

        if do_detection:
            detector_ok = self.detector is not None and self.detector.available
            pose_ok = self.pose is not None and self.pose.enabled and self.pose.available
            # ===== MEGA BATCH: central GPU call =====
            if detector_ok:
                if self.batch_scheduler is not None:
                    self._detection_event.clear()
                    self.batch_scheduler.submit(
                        self.camera_id, frame, fid, self._on_batch_detections
                    )
                    if self._detection_event.wait(timeout=0.10):
                        with self._detection_lock:
                            boxes = self._detection_result or []
                    else:
                        # Timeout bo'lsa tracker bashorat bilan silliq davom etadi (GPU qotmasligi uchun)
                        boxes = []
                else:
                    boxes = self.detector.detect(frame)
                kpts = [None] * len(boxes)
            elif pose_ok:
                boxes, kpts = self.pose.detect(frame)
            else:
                log.warning("No detection engine available for %s", self.camera_id)


        # ---------------- Stage 2: Tracking ----------------
        if do_detection:
            tracks = self.tracker.update(boxes)
            self._last_tracks = tracks
        else:
            tracks = getattr(self, "_last_tracks", [])

        # ---------------- Stage 3: Face Recognition ----------------
        if do_detection and tracks:
            self._process_faces(frame, tracks, fid)

        # ---------------- Stage 4: ReID ----------------
        if do_detection and self.reid is not None and self.reid.enabled and self.reid.available:
            self._process_reid(frame, tracks, fid)

        # Event source is person detection/tracking; face only enriches identity.
        if do_detection:
            for tr in tracks:
                self._emit_person_detected(frame, tr)

        # ---------------- Stage 6: Track termination ----------------
        self._cleanup_cache(tracks)

        # ---------------- Ankle map (heatmap uchun) ----------------
        ankle_map = self._build_ankle_map(boxes, kpts, tracks)

        # ---------------- Build persons ----------------
        persons = self._build_persons(tracks, ankle_map)

        return persons

    # ---------------- ankle / heatmap ----------------
    def _build_ankle_map(self, boxes, kpts, tracks):
        ankle_map = {}

        if kpts is not None and len(kpts) > 0 and self.pose is not None:
            for tr in tracks:
                best_ankle = None
                best_iou = 0.3

                for box, kpt in zip(boxes, kpts):
                    if kpt is None:
                        continue

                    iou_val = iou(tr.box, box)

                    if iou_val > best_iou:
                        ankle = self.pose.ankle_point(kpt)

                        if ankle is not None:
                            best_ankle = ankle
                            best_iou = iou_val

                if best_ankle is None:
                    _b = tr.box
                    if _b is not None and len(_b) >= 4:
                        x1, y1, x2, y2 = float(_b[0]), float(_b[1]), float(_b[2]), float(_b[3])
                        best_ankle = ((x1 + x2) / 2.0, y2)

                ankle_map[tr.id] = best_ankle

        else:
            # Pose yo‘q bo‘lsa, bbox pastki markazi ankle sifatida ishlatiladi
            for tr in tracks:
                _b = tr.box
                if _b is None or len(_b) < 4:
                    continue
                x1, y1, x2, y2 = float(_b[0]), float(_b[1]), float(_b[2]), float(_b[3])
                ankle_map[tr.id] = ((x1 + x2) / 2.0, y2)

        return ankle_map

    # ---------------- face recognition ----------------
    def _process_faces(self, frame, tracks, fid):
        self._current_track_ids = set(tr.id for tr in tracks)   # zona qulfi uchun
        if self.face is None or not self.face.enabled or not self.face.available:
            return

        now = time.time()
        need_face_tracks = []

        for tr in tracks:
            cache = self.identity_cache.get(tr.id)

            if cache is None:
                # Stage 5: yangi Track ID → face recognition kerak
                need_face_tracks.append(tr)

            elif cache["person_id"] is None:
                # hali tanilmagan → timeout bilan qayta urinamiz
                if now - cache["last_recognition_time"] > self.face_timeout:
                    need_face_tracks.append(tr)

            else:
                # allaqachon tanilgan
                if cache.get("source") == "reid":
                    # 🔁 ReID TAXMINI → face bilan tez tasdiqlash (xatoni tuzatish!)
                    if now - cache["last_recognition_time"] > self.face_timeout:
                        need_face_tracks.append(tr)
                else:
                    # ✅ face bilan tasdiqlangan → sekin revalidate
                    if now - cache["last_recognition_time"] > self.face_timeout * 5:
                        need_face_tracks.append(tr)

        if not need_face_tracks:
            return

        # Start retry timeout even if the person is back-facing/no face is visible.
        # Otherwise InsightFace is invoked on every detection cycle.
        for tr in need_face_tracks:
            if self.identity_cache.get(tr.id) is None:
                self.identity_cache.set(
                    tr.id, None, f"Unknown-{tr.id}", None, 0.0,
                    self.camera_id, fid,
                )

        faces = self.face.detect(frame)

        # Distant-face fallback: scan the upper body crop only when the global
        # frame scan did not find a face inside that person track.
        covered = set()
        for face in faces:
            matched = self._match_face_to_track(face, need_face_tracks)
            if matched is not None:
                covered.add(matched.id)
        
        # Throttled distant-face fallback to preserve GPU resources
        if fid % 10 == 0:
            h, w = frame.shape[:2]
            fallback_tracks = [tr for tr in need_face_tracks if tr.id not in covered]
            fallback_tracks.sort(key=lambda tr: (tr.box[2]-tr.box[0])*(tr.box[3]-tr.box[1]), reverse=True)
            for tr in fallback_tracks[:1]:
                x1, y1, x2, y2 = [int(v) for v in tr.box[:4]]
                bw, bh = x2-x1, y2-y1
                if bw < 24 or bh < 60:
                    continue
                px1 = max(0, x1-int(bw*0.12)); px2 = min(w, x2+int(bw*0.12))
                py1 = max(0, y1-int(bh*0.08)); py2 = min(h, y1+int(bh*0.62))
                crop = frame[py1:py2, px1:px2]
                if crop.size == 0:
                    continue
                local_faces = self.face.detect(crop)
                if not local_faces:
                    continue
                local = max(local_faces, key=lambda f: (f["bbox"][2]-f["bbox"][0])*(f["bbox"][3]-f["bbox"][1]))
                local["bbox"] = [local["bbox"][0]+px1, local["bbox"][1]+py1,
                                 local["bbox"][2]+px1, local["bbox"][3]+py1]
                faces.append(local)

        if not faces:
            return

        for face in faces:
            tr = self._match_face_to_track(face, need_face_tracks)

            if tr is None:
                continue

            try:
                q = self.face.quality(frame, face)
            except Exception as _qe:
                q = {"score": 100.0}  # quality xato bo'lsa, o'tkazish

            if q.get("score", 0) < self.face_min_quality:
                cache = self.identity_cache.get(tr.id)

                if cache is None:
                    self.identity_cache.set(
                        tr.id,
                        None,
                        f"Unknown-{tr.id}",
                        None,
                        0.0,
                        self.camera_id,
                        fid,
                    )

                continue

            # Batched face recognition (vectorized matrix dot product)
            if hasattr(self.face, "recognize_batch"):
                res_rec = self.face.recognize_batch([face["embedding"]])[0]
                person_id, name, score = res_rec
            else:
                person_id, name, score = self.face.recognize(face["embedding"])

            if person_id is not None and score >= self.face.threshold:
                # ✅ MUHIM: odam tanilganda last_seen yangilanadi
                try:
                    if hasattr(self, 'sm') and hasattr(self.sm, 'db'):
                        self.sm.db.update_person_last_seen(person_id, 1)
                except Exception:
                    pass

            if person_id is not None and score >= self.face.threshold:
                # Tanildi → cache
                self.identity_cache.set(
                    tr.id,
                    person_id,
                    name,
                    face["embedding"],
                    score,
                    self.camera_id,
                    fid,
                )

                # 🔁 ReID uchun tana feature saqlash (keyin yuz ko'rinmasa ham tanish)
                self._store_body_feature(frame, tr, person_id, name, score)
                self._maybe_save_avatar(frame, face, person_id, q.get('score', 0))

            else:
                # Unknown face embedding receives one thread-safe global ID,
                # so the same face has the same label on every camera.
                _uname = f"Unknown-{tr.id}"
                if self.unknown_registry is not None and face.get("embedding") is not None:
                    try:
                        _uname, _ = self.unknown_registry.match_or_create(face["embedding"])
                    except Exception as exc:
                        log.debug("Unknown face registry error: %s", exc)
                self.identity_cache.set(
                    tr.id, None, _uname, face["embedding"], score,
                    self.camera_id, fid,
                )
                try:
                    self.identity_cache.cache[tr.id]["source"] = "face_unknown"
                except Exception:
                    pass

    def _emit_person_detected(self, frame, tr):
        if int(getattr(tr, "hits", 0)) < 2:
            return
        cache = self.identity_cache.get(tr.id)
        # ✅ TOZA FORMAT: unk:ID yoki ism:ID
        cached_name = (cache or {}).get("name")
        if not cached_name or cached_name.startswith("Person-") or cached_name.startswith("Unknown-"):
            unk_id = (cache or {}).get("unknown_id") or f"UNK-{tr.id}"
            name = f"unk:{unk_id}"
        else:
            p_id = (cache or {}).get("person_id") or tr.id
            name = f"{cached_name}:{p_id}"
        person_id = (cache or {}).get("person_id")
        self._emit_dedup(
            frame, tr, person_id, name, float(getattr(tr, "conf", 0.0) or 0.0),
            "person_detected", "info", {"source": "person_detector", "track_id": tr.id},
        )

    def _maybe_save_avatar(self, frame, face, person_id, quality=0.0):
        """Yuz crop ini avatar sifatida saqlash (verbose diagnostika bilan)."""
        if not hasattr(self, "_avatar_done"):
            self._avatar_done = set()
        if person_id in self._avatar_done:
            return
        try:
            sm = getattr(self, "sm", None)
            if sm is None or not hasattr(sm, "db"):
                print(f"[AVATAR] sm/db yo'q", flush=True); return
            with sm.db.lock:
                row = sm.db.conn.execute("SELECT length(avatar) AS l FROM persons WHERE id=?", (person_id,)).fetchone()
            if row and row["l"] and row["l"] > 100:
                self._avatar_done.add(person_id); return
            box = None
            for key in ("bbox", "box", "face", "rect", "location", "bounding_box"):
                v = face.get(key) if isinstance(face, dict) else None
                if v is not None and len(v) >= 4:
                    box = [int(x) for x in v[:4]]; break
            if box is None:
                print(f"[AVATAR] box topilmadi! face keys={list(face.keys()) if isinstance(face, dict) else type(face)}", flush=True); return
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = box
            fw, fh = max(1, x2 - x1), max(1, y2 - y1)
            x1 = max(0, x1 - int(fw * 0.25)); y1 = max(0, y1 - int(fh * 0.40))
            x2 = min(w, x2 + int(fw * 0.25)); y2 = min(h, y2 + int(fh * 0.20))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"[AVATAR] crop bo'sh (box={box})", flush=True); return
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                print(f"[AVATAR] encode xato", flush=True); return
            img = buf.tobytes()
            with sm.db.lock:
                sm.db.conn.execute("UPDATE persons SET avatar=? WHERE id=?", (img, person_id))
                sm.db.conn.commit()
            self._avatar_done.add(person_id)
            print(f"[AVATAR] SAQLANDI: person_id={person_id} ({len(img)} bayt)", flush=True)
        except Exception as e:
            print(f"[AVATAR] XATO: {e}", flush=True)

    def _emit_dedup(self, frame, tr, person_id, name, score, etype, level, extra=None):
        """DEDUP + ZONA (soddalashtirilgan, SPAM GA QARSHI).
        - Faol track qulfi OLINDI (u spam qilardi: Husan turgan joyda yangi track=unknown).
        - Zona: unknown box i tanilgan odam zonasiga tushsa -> unknown BEKOR (egilish/yuz yopiq).
        - Box-zona dedup: bir joyda 8s ichida takror unknown YO'Q -> spam to'xtaydi.
        """
        import time as _t2
        _now2 = _t2.time()
        # ✅ Global dedup OLIB TASHLANDI — faqat kamera+track based dedup ishlaydi
        # ✅ KAMERA-BASED DEDUP: bir kamera+person = 30s ichida bir marta
        # Boshqa kameraga o'tsa → yangi event (cross-camera)
        key = (etype, self.camera_id, tr.id)
        _cur = tr.box[:4] if (tr.box is not None and len(tr.box) >= 4) else None
        _cx = _cy = None
        if _cur is not None:
            _cx = (float(_cur[0]) + float(_cur[2])) / 2.0
            _cy = (float(_cur[1]) + float(_cur[3])) / 2.0

        # ZONA tekshiruvi (tanilgan odam zonasi)
        if _cur is not None:
            for _z in self._recent_recognized:
                _rb, _rpid, _rname = _z[0], _z[1], _z[2]
                _bw = max(1.0, float(_rb[2]) - float(_rb[0]))
                _bh = max(1.0, float(_rb[3]) - float(_rb[1]))
                _zx = (float(_rb[0]) + float(_rb[2])) / 2.0
                _zy = (float(_rb[1]) + float(_rb[3])) / 2.0
                _dx = abs(_cx - _zx); _dy = abs(_cy - _zy)
                if _dx < max(60.0, _bw * 0.6) and _dy < _bh * 1.3:
                    if etype == "unknown_detected":
                        try:
                            self.identity_cache.set(tr.id, _rpid, _rname, None, 0.0, self.camera_id, -1)
                            self.identity_cache.cache[tr.id]["source"] = "zone"
                        except Exception:
                            pass
                        print(f"[AIWorker {self.camera_id}] ZONE TRACK: {_rname} (dx={_dx:.0f} dy={_dy:.0f}) - unknown BEKOR", flush=True)
                        return
                    if etype == "person_recognized":
                        _z[0] = list(_cur)
                    break

            if len(self._unknown_zone_ts) > 100:
                self._unknown_zone_ts = self._unknown_zone_ts[-100:]

        # Oddiy dedup (bir track+person = bir marta)
        if key in self._emitted:
            return
        self._emitted.add(key)
        if len(self._emitted) > 2000:
            self._emitted = set(list(self._emitted)[-1000:])

        # recognized -> zona ga yozish
        if etype == "person_recognized" and _cur is not None:
            _found = False
            for _z in self._recent_recognized:
                if _z[1] == person_id:
                    _z[0] = list(_cur); _found = True; break
            if not _found:
                self._recent_recognized.append([list(_cur), person_id, name])
            if len(self._recent_recognized) > 300:
                self._recent_recognized = self._recent_recognized[-300:]

        _snap = self._save_person_crop(frame, tr.box, self.camera_id, name)
        payload = {"type": etype, "level": level, "camera_id": self.camera_id,
                   "person_id": person_id, "person_name": name, "confidence": score,
                   "snapshot_path": _snap}
        if extra:
            payload.update(extra)
        self.event_detected.emit(self.camera_id, payload)
        print(f"[AIWorker {self.camera_id}] EVENT {etype}: {name} track={tr.id}", flush=True)


    def _save_person_crop(self, frame, box, camera_id, label):
        """Odam box ni frame dan crop qilib saqlash (butun frame emas).
        Returns: snapshot path yoki None"""
        try:
            if frame is None or box is None or len(box) < 4:
                return None
            import cv2, os
            from datetime import datetime as _dt
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            _now = _dt.now()
            date_folder = _now.strftime("%d-%m-%Y")          # 📁 papka: 29-07-2026 (sana)
            snap_dir = os.path.join("data", "snapshots", date_folder)
            os.makedirs(snap_dir, exist_ok=True)
            time_str = _now.strftime("%H-%M-%S")             # 🕐 fayl: 16-50-34 (vaqt, ms yo'q)
            safe = "".join(c if c.isalnum() else "_" for c in str(label))[:20]
            path = os.path.join(snap_dir, f"{camera_id}_{safe}_{time_str}.jpg")
            cv2.imwrite(path, crop)
            return path
        except Exception as _ce:
            print(f"[AIWorker {self.camera_id}] crop error: {_ce}", flush=True)
            return None

    def _match_face_to_track(self, face, tracks):
        """Yuzni track ga bog'lash: yuz markazi track body box ichida bo'lsa.
        IoU emas (yuz << tana, IoU juda kichik bo'ladi), containment ishlatiladi."""
        fx1, fy1, fx2, fy2 = face["bbox"]
        fcx = (fx1 + fx2) / 2.0
        fcy = (fy1 + fy2) / 2.0

        best = None
        best_area = float("inf")

        for tr in tracks:
            _box = tr.box
            if _box is None or len(_box) < 4:
                continue
            x1, y1, x2, y2 = float(_box[0]), float(_box[1]), float(_box[2]), float(_box[3])

            # Yuz markazi track body box ichidami?
            if x1 <= fcx <= x2 and y1 <= fcy <= y2:
                area = (x2 - x1) * (y2 - y1)
                # Eng kichik (eng zich) box — eng yaqin track
                if area < best_area:
                    best_area = area
                    best = tr

        return best

    def _crop_body(self, frame, box):
        """Track box dan tana crop olish"""
        if frame is None or box is None or len(box) < 4:
            return None
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _store_body_feature(self, frame, tr, person_id, name, face_score=1.0):
        """Yuz tanilganda tana ReID feature ni gallery ga saqlash"""
        if face_score < self.face.threshold:
            return  # ✅ faqat SIFATLI face dan gallery (false positive kamayadi)
        try:
            if self.reid is None or not getattr(self.reid, "available", False):
                return
            crop = self._crop_body(frame, tr.box)
            if crop is None:
                return
            feat = self.reid.extract_features(crop)
            if feat is None:
                return
            gal = self.body_gallery.setdefault(person_id, [])
            gal.append(feat)
            if len(gal) > 5:
                gal.pop(0)
            self.body_gallery_names[person_id] = name
            self.track_body_features[tr.id] = feat
            if self.shared_gallery is not None:
                try:
                    self.shared_gallery.add(person_id, name, feat, camera_id=self.camera_id, room=self.room)
                except Exception:
                    pass
        except Exception as e:
            print(f"[AIWorker {self.camera_id}] store_body error: {e}", flush=True)

    def _process_reid(self, frame, tracks, fid):
        """Cross-camera ReID: 2-pass (known) + Unknown tracking (unknown_id)."""
        if self.reid is not None and not getattr(self.reid, "available", False):
            return
        _has_shared = (self.shared_gallery is not None and self.shared_gallery.size() > 0)

        def _crop_safe(box):
            if box is None or len(box) < 4: return None
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w, x2), min(h, y2)
            c = frame[y1:y2, x1:x2]
            return c if c.size > 0 else None

        # 1-PASS: barcha track lar uchun batch ReID feature extraction (GPU Batched)
        candidates = []
        pending_tracks = []
        pending_crops = []

        reid_now = time.time()
        for tr in tracks:
            cache = self.identity_cache.get(tr.id)
            if cache is not None and cache.get("person_id") is not None:
                continue  # known, skip
            if reid_now - self._reid_last_ts.get(tr.id, 0.0) < 1.0:
                continue
            self._reid_last_ts[tr.id] = reid_now

            crop = _crop_safe(tr.box)
            if crop is None:
                continue

            pending_tracks.append(tr)
            pending_crops.append(crop)

        if not pending_tracks:
            return

        # Batched feature extraction on GPU
        if hasattr(self.reid, "extract_features_batch"):
            feats = self.reid.extract_features_batch(pending_crops)
        else:
            feats = [self.reid.extract_features(c) for c in pending_crops]

        for tr, feat in zip(pending_tracks, feats):
            if feat is None:
                continue

            self.track_body_features[tr.id] = feat

            # Known match (shared_gallery + body_gallery)
            _gs = self.shared_gallery.size() if self.shared_gallery else 0
            _eff = self.reid_threshold
            if _gs <= 1: _eff = max(_eff, 0.70)
            elif _gs <= 3: _eff = max(_eff, 0.68)

            best_pid, best_name, best_score = None, None, 0.0
            if self.shared_gallery is not None and _gs > 0:
                best_pid, best_name, best_score = self.shared_gallery.match(feat, _eff, self.camera_id, tr.id)
            if best_pid is None:
                for pid, feats in self.body_gallery.items():
                    for gf in feats:
                        sim = self.reid.compute_similarity(feat, gf)
                        if sim > best_score:
                            best_score = sim; best_pid = pid
                            best_name = self.body_gallery_names.get(pid, f"Person-{pid}")
                if best_score < _eff:
                    best_pid = None

            if best_pid is not None and best_score >= _eff:
                candidates.append((best_score, tr, best_pid, best_name, _eff))
                continue

            # UNKNOWN MATCH: shared and locked across all camera workers.
            with type(self)._shared_unknown_lock:
                best_uid, best_uscore = None, 0.0
                now_reid = time.time()
                reservations = type(self)._shared_unknown_reservations
                reservations_copy = dict(reservations)
                for uid, feats in self.unknown_gallery.items():
                    held = reservations_copy.get((uid, self.camera_id))
                    if held and held[0] == self.camera_id and held[1] != tr.id and now_reid - held[2] < 3.0:
                        continue
                    for gf in feats:
                        sim = self.reid.compute_similarity(feat, gf)
                        if sim > best_uscore:
                            best_uscore, best_uid = sim, uid

                current = self.identity_cache.get(tr.id)
                face_uid = (current or {}).get("name") if (current or {}).get("source") == "face_unknown" else None
                face_held = reservations.get((face_uid, self.camera_id)) if face_uid else None
                if face_uid and not (face_held and face_held[0] == self.camera_id and face_held[1] != tr.id and now_reid - face_held[2] < 3.0):
                    unknown_id = face_uid
                elif best_uid is not None and best_uscore > self.reid_threshold:
                    unknown_id = best_uid
                else:
                    if self.unknown_registry is not None:
                        unknown_id, _ = self.unknown_registry.match_or_create(None)
                    else:
                        unknown_id = f"UNK-{type(self)._shared_unknown_next_id}"
                        type(self)._shared_unknown_next_id += 1
                    best_uscore = 0.0
                self.unknown_gallery.setdefault(unknown_id, []).append(feat)
                self.unknown_gallery[unknown_id] = self.unknown_gallery[unknown_id][-10:]
                reservations[(unknown_id, self.camera_id)] = (self.camera_id, tr.id, now_reid)
                for rid, held in list(reservations.items()):
                    if now_reid - held[2] > 15.0:
                        reservations.pop(rid, None)

            # identity_cache ga yozish (track davom etsin)
            self.identity_cache.set(tr.id, None, str(unknown_id), None, best_uscore, self.camera_id, fid)
            try:
                self.identity_cache.cache[tr.id]["unknown_id"] = unknown_id
                self.identity_cache.cache[tr.id]["source"] = "unknown_reid"
            except Exception:
                pass

            if self.frame_count % 30 == 1:
                print(f"[AIWorker {self.camera_id}] UNKNOWN: track={tr.id} unknown_id={unknown_id} score={best_uscore:.3f}", flush=True)

        # 2-PASS: per-person eng yuqori score (known)
        best_per_person = {}
        for score, tr, pid, name, _eff in candidates:
            if pid not in best_per_person or score > best_per_person[pid][0]:
                best_per_person[pid] = (score, tr, name, _eff)

        # 3-PASS: faqat eng yaxshi match larni cache ga yozish (known)
        for pid, (score, tr, name, _eff) in best_per_person.items():
            import time as _t3
            _now3 = _t3.time()
            _last_cam, _last_room, _last_seen = None, None, 0.0
            if self.shared_gallery is not None:
                try:
                    _last_cam, _last_room = self.shared_gallery.get_location(pid)
                    _last_seen = self.shared_gallery.last_seen.get(pid, 0.0)
                except Exception:
                    pass
            _dt = (_now3 - _last_seen) if _last_seen else 9999.0
            if _last_room is not None and _last_room != self.room and _dt < self.min_transition:
                print(f"[AIWorker {self.camera_id}] REID REJECT: {name} turli xona dt={_dt:.1f}s -> false", flush=True)
                continue
            self.identity_cache.set(tr.id, pid, name, None, score, self.camera_id, fid)
            try:
                self.identity_cache.cache[tr.id]["source"] = "reid"
            except Exception:
                pass
            # ✅ _emit_dedup OLIB TASHLANDI (chunki _process_faces allaqachon event emit qiladi)
            # self._emit_dedup(frame, tr, pid, name, score, "person_recognized", "ok", {"source": "reid"})
            print(f"[AIWorker {self.camera_id}] ReID MATCH: {name} score={score:.3f} eff={_eff:.2f} track={tr.id} (event yo'q, faqat cache)", flush=True)


    def _cleanup_cache(self, tracks):
        # Keep identity while ByteTracker keeps a lost track recoverable.
        tracker_tracks = getattr(self.tracker, "tracks", tracks)
        active_ids = {tr.id for tr in tracker_tracks
                      if getattr(tr, "missing", 0) <= getattr(self.tracker, "track_buffer", 30)}

        for tid in list(self.identity_cache.active_ids()):
            if tid not in active_ids:
                self.identity_cache.remove(tid)
                self._reid_last_ts.pop(tid, None)

    # ---------------- build persons ----------------
    def _build_persons(self, tracks, ankle_map):
        persons = []

        for tr in tracks:
            cache = self.identity_cache.get(tr.id)
            # 🧠 ZONA (yuzsiz online): cache da yo'q bo'lsa ham, box tanilgan zonasiga tushsa -> Husan
            if (cache is None or cache.get("person_id") is None) and getattr(tr, "box", None) is not None and len(tr.box) >= 4:
                try:
                    _cx = (float(tr.box[0]) + float(tr.box[2])) / 2.0
                    _cy = (float(tr.box[1]) + float(tr.box[3])) / 2.0
                    for _z in getattr(self, "_recent_recognized", []):
                        _rb = _z[0]
                        _bw = max(1.0, float(_rb[2]) - float(_rb[0]))
                        _bh = max(1.0, float(_rb[3]) - float(_rb[1]))
                        _zx = (float(_rb[0]) + float(_rb[2])) / 2.0
                        _zy = (float(_rb[1]) + float(_rb[3])) / 2.0
                        if abs(_cx - _zx) < max(60.0, _bw * 0.6) and abs(_cy - _zy) < _bh * 1.3:
                            cache = {"person_id": _z[1], "name": _z[2], "confidence": 0.0, "source": "zone"}
                            break
                except Exception:
                    pass

            if cache is not None:
                known = cache["person_id"] is not None
                name = cache["name"]
                person_id = cache["person_id"]
                face_conf = cache["confidence"]
            else:
                known = False
                name = f"UNK-{tr.id}"
                person_id = None
                face_conf = 0.0

            p = DetectedPerson(
                track_id=tr.id,
                box=list(tr.box),
                conf=tr.conf,
                known=known,
                person_id=person_id,
                name=name,
                face_conf=face_conf,
                ankle=ankle_map.get(tr.id),
                age=time.time() - tr.first_seen,
                first_seen=tr.first_seen,
            )

            persons.append(p)

        return persons