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
                persons = self._run_pipeline(frame, fid)
                self.last_persons = persons
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
                    x1, y1, x2, y2 = tr.box
                    best_ankle = ((x1 + x2) / 2.0, y2)

                ankle_map[tr.id] = best_ankle

        else:
            # Pose yo‘q bo‘lsa, bbox pastki markazi ankle sifatida ishlatiladi
            for tr in tracks:
                x1, y1, x2, y2 = tr.box
                ankle_map[tr.id] = ((x1 + x2) / 2.0, y2)

        return ankle_map

    # ---------------- face recognition ----------------
    def _process_faces(self, frame, tracks, fid):
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
                # allaqachon tanilgan → faqat uzoq vaqtdan keyin revalidate
                if now - cache["last_recognition_time"] > self.face_timeout * 5:
                    need_face_tracks.append(tr)

        if not need_face_tracks:
            return

        faces = self.face.detect(frame)

        if not faces:
            return

        for face in faces:
            tr = self._match_face_to_track(face, need_face_tracks)

            if tr is None:
                continue

            q = self.face.quality(frame, face)

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

                self.event_detected.emit(
                    self.camera_id,
                    {
                        "type": "person_recognized",
                        "level": "ok",
                        "camera_id": self.camera_id,
                        "person_id": person_id,
                        "person_name": name,
                        "confidence": score,
                    },
                )

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

                self.event_detected.emit(
                    self.camera_id,
                    {
                        "type": "unknown_detected",
                        "level": "warn",
                        "camera_id": self.camera_id,
                        "person_name": f"Unknown-{tr.id}",
                        "confidence": score,
                    },
                )

    def _match_face_to_track(self, face, tracks):
        fx1, fy1, fx2, fy2 = face["bbox"]

        fcx = (fx1 + fx2) / 2.0
        fcy = (fy1 + fy2) / 2.0

        best = None
        best_iou = 0.1

        for tr in tracks:
            x1, y1, x2, y2 = tr.box

            if x1 <= fcx <= x2 and y1 <= fcy <= y2:
                s = iou(tr.box, face["bbox"])

                if s > best_iou:
                    best_iou = s
                    best = tr

        return best

    # ---------------- reid ----------------
    def _process_reid(self, frame, tracks, fid):
        """
        ReID hozircha arxitektura uchun tayyor.
        Face ko‘rinmasa va Track ID yo‘qolmasa, cached identity ishlatiladi.
        """
        pass

    # ---------------- track termination ----------------
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