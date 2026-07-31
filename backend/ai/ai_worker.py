import time

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
        self._emitted = {}              # ✅ dedup: (track_id,person_id)→name
        self.shared_gallery = None      # 🌐 cross-camera gallery (ServiceManager set qiladi)
        self.body_gallery = {}          # person_id → [HSV histogram feature]
        self.unknown_gallery = {}       # ✅ unknown_id → [feature] (unknown ReID tracking)
        self._next_unknown_id = 1       # ✅ keyingi unknown_id
        self._current_track_ids = set()   # hozirgi frame dagi barcha track id
        self._recent_recognized = []   # 🧠 [(box4,person_id,name,time)] yuz burilgan odam xotirasi
        self.body_gallery_names = {}    # person_id → name
        self.track_body_features = {}   # track_id → oxirgi body feature
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

    def stop(self):
        self._running = False

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

            # Har 10 frame da debug
            if self.frame_count % 10 == 1:
                print(f"[AIWorker {self.camera_id}] FRAME #{self.frame_count} fid={fid} shape={getattr(frame,'shape','?')}", flush=True)

            run_detection = (self.frame_count % self.detection_interval == 0)

            if run_detection:
                try:
                    persons = self._run_pipeline(frame, fid)
                    self.last_persons = persons
                except Exception as _pe:
                    # ✅ GLOBAL FIX: thread hech qachon o'lmaydi
                    print(f"[AIWorker {self.camera_id}] ⚠ pipeline xato (thread yashaydi): {_pe}", flush=True)
                    import traceback as _tb; _tb.print_exc()
                    persons = self.last_persons
            else:
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
    def _run_pipeline(self, frame, fid):
        # ---------------- Stage 1: Detection ----------------
        kpts = None

        if self.pose is not None and self.pose.enabled and self.pose.available:
            boxes, kpts = self.pose.detect(frame)
            print(f"[AIWorker {self.camera_id}] POSE DETECT: {len(boxes)} boxes", flush=True)
        elif self.detector is not None and self.detector.available:
            boxes = self.detector.detect(frame)
            print(f"[AIWorker {self.camera_id}] DET DETECT: {len(boxes)} boxes", flush=True)
        else:
            boxes = []
            print(f"[AIWorker {self.camera_id}] NO ENGINE AVAILABLE!", flush=True)

        # ---------------- Stage 2: Tracking ----------------
        # ✅ DEBUG: detection conf qiymatlarini ko'rsatish
        if boxes:
            confs = [b[4] if len(b) > 4 else -1 for b in boxes]
            print(f"[AIWorker {self.camera_id}] BOX CONFS: {[round(c, 3) for c in confs]}", flush=True)
        
        tracks = self.tracker.update(boxes)
        print(f"[AIWorker {self.camera_id}] TRACKS: {len(tracks)} (thresh={self.tracker.new_track_thresh})", flush=True)

        # ---------------- Stage 3: Face Recognition ----------------
        self._process_faces(frame, tracks, fid)

        # ---------------- Stage 4: ReID ----------------
        if self.reid is not None and self.reid.enabled and self.reid.available:
            self._process_reid(frame, tracks, fid)

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

        if self.frame_count % 15 == 1:
            print(f"[AIWorker {self.camera_id}] 🎯 FACE-STAGE: tracks={len(tracks)} need_face={len(need_face_tracks)}", flush=True)

        if not need_face_tracks:
            return

        faces = self.face.detect(frame)
        if self.frame_count % 15 == 1:
            print(f"[AIWorker {self.camera_id}] 🔍 FACE-DETECT: raw={len(faces) if faces else 0}", flush=True)

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

            person_id, name, score = self.face.recognize(face["embedding"])
            if self.frame_count % 15 == 1:
                print(f"[AIWorker {self.camera_id}] 🧠 RECOGNIZE: pid={person_id} name={name} score={score:.3f} thresh={self.face.threshold}", flush=True)

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

                self._emit_dedup(frame, tr, person_id, name, score, "person_recognized", "ok")

            else:
                # Tanilmadi → unknown cache
                self.identity_cache.set(
                    tr.id,
                    None,
                    f"Unknown-{tr.id}",
                    face["embedding"],
                    score,
                    self.camera_id,
                    fid,
                )

                self._emit_dedup(frame, tr, None, f"Unknown-{tr.id}", score, "unknown_detected", "warn")

    def _emit_dedup(self, frame, tr, person_id, name, score, etype, level, extra=None):
        """DEDUP + ZONA (soddalashtirilgan, SPAM GA QARSHI).
        - Faol track qulfi OLINDI (u spam qilardi: Husan turgan joyda yangi track=unknown).
        - Zona: unknown box i tanilgan odam zonasiga tushsa -> unknown BEKOR (egilish/yuz yopiq).
        - Box-zona dedup: bir joyda 8s ichida takror unknown YO'Q -> spam to'xtaydi.
        """
        import time as _t2
        _now2 = _t2.time()
        key = (tr.id, person_id)
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

        # BOX-ZONA DEDUP (spam to'xtatuvchi): bir joyda 8s ichida takror unknown YO'Q
        if etype == "unknown_detected" and _cx is not None:
            if not hasattr(self, "_unknown_zone_ts"):
                self._unknown_zone_ts = []
            self._unknown_zone_ts = [z for z in self._unknown_zone_ts if _now2 - z[2] < 8.0]
            for _ux, _uy, _ut in self._unknown_zone_ts:
                if ((_cx - _ux) ** 2 + (_cy - _uy) ** 2) ** 0.5 < 120.0:
                    return  # shu joyda yaqinda unknown chiqqan -> takror YO'Q
            self._unknown_zone_ts.append((_cx, _cy, _now2))
            if len(self._unknown_zone_ts) > 100:
                self._unknown_zone_ts = self._unknown_zone_ts[-100:]

        # Oddiy dedup (bir track+person = bir marta)
        if self._emitted.get(key) == name:
            return
        self._emitted[key] = name
        if len(self._emitted) > 500:
            for k in list(self._emitted.keys())[:-250]:
                self._emitted.pop(k, None)

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
        if face_score < 0.6:
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
        if self.frame_count % 30 == 1:
            print(f"[AIWorker {self.camera_id}] REID-STAGE: tracks={len(tracks)} shared={self.shared_gallery.size() if self.shared_gallery else 0} local={len(self.body_gallery)} unknown={len(self.unknown_gallery)}", flush=True)
        if not self.body_gallery and not _has_shared and not self.unknown_gallery:
            return

        def _crop_safe(box):
            if box is None or len(box) < 4: return None
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w, x2), min(h, y2)
            c = frame[y1:y2, x1:x2]
            return c if c.size > 0 else None

        # 1-PASS: barcha track lar uchun match (known + unknown)
        candidates = []
        for tr in tracks:
            cache = self.identity_cache.get(tr.id)
            if cache is not None and cache.get("person_id") is not None:
                continue  # known, skip

            crop = _crop_safe(tr.box)
            if crop is None:
                continue

            feat = self.reid.extract_features(crop)
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
                best_pid, best_name, best_score = self.shared_gallery.match(feat, _eff)
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

            # ✅ UNKNOWN MATCH (unknown_gallery)
            best_uid, best_uscore = None, 0.0
            for uid, feats in self.unknown_gallery.items():
                for gf in feats:
                    sim = self.reid.compute_similarity(feat, gf)
                    if sim > best_uscore:
                        best_uscore = sim
                        best_uid = uid

            if best_uid is not None and best_uscore > 0.65:
                unknown_id = best_uid
                self.unknown_gallery[unknown_id].append(feat)
                if len(self.unknown_gallery[unknown_id]) > 10:
                    self.unknown_gallery[unknown_id] = self.unknown_gallery[unknown_id][-10:]
            else:
                unknown_id = self._next_unknown_id
                self._next_unknown_id += 1
                self.unknown_gallery[unknown_id] = [feat]
                best_uscore = 0.0

            # identity_cache ga yozish (track davom etsin)
            self.identity_cache.set(tr.id, None, f"Unknown-{unknown_id}", None, best_uscore, self.camera_id, fid)
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
            self._emit_dedup(frame, tr, pid, name, score, "person_recognized", "ok", {"source": "reid"})
            print(f"[AIWorker {self.camera_id}] ReID MATCH: {name} score={score:.3f} eff={_eff:.2f} track={tr.id}", flush=True)


    def _cleanup_cache(self, tracks):
        active_ids = {tr.id for tr in tracks}

        for tid in list(self.identity_cache.active_ids()):
            if tid not in active_ids:
                self.identity_cache.remove(tid)

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
                name = f"Unknown-{tr.id}"
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