"""
Global Person Pool — barcha kameralar uchun umumiy shaxs havzasi.
Cross-camera tracking ni ta'minlaydi.
"""
import threading
import time
from typing import Optional, Dict, Tuple
import numpy as np

class Person:
    """Global shaxs — barcha kameralarda bitta"""
    def __init__(self, person_id: int, name: str = None, is_known: bool = False):
        self.person_id = person_id
        self.name = name or f"Person-{person_id}"
        self.is_known = is_known
        
        # Yuz embedding (agar bor bo'lsa)
        self.face_emb: Optional[np.ndarray] = None
        self.face_conf: float = 0.0
        
        # Body histogram (Re-ID uchun)
        self.body_hist: Optional[np.ndarray] = None
        
        # Tracking ma'lumotlari
        self.last_seen: float = time.time()
        self.last_camera: Optional[str] = None
        self.last_track_id: Optional[int] = None
        
        # Statistika
        self.detection_count: int = 0
        self.cameras_seen: set = set()
        
        # Crop rasmlar (debug uchun)
        self.best_face_crop = None
        self.best_body_crop = None
    
    def update_from_face(self, face_emb: np.ndarray, conf: float, 
                         camera_id: str, track_id: int, crop=None):
        """Yuz orqali yangilash"""
        self.face_emb = face_emb
        self.face_conf = conf
        self.last_seen = time.time()
        self.last_camera = camera_id
        self.last_track_id = track_id
        self.detection_count += 1
        self.cameras_seen.add(camera_id)
        if crop is not None:
            self.best_face_crop = crop
    
    def update_from_body(self, body_hist: np.ndarray, 
                         camera_id: str, track_id: int, crop=None):
        """Body orqali yangilash"""
        # Agar body_hist yaxshiroq bo'lsa, yangilash
        if self.body_hist is None:
            self.body_hist = body_hist
        else:
            # Exponential moving average (eski 70%, yangi 30%)
            self.body_hist = self.body_hist * 0.7 + body_hist * 0.3
        
        self.last_seen = time.time()
        self.last_camera = camera_id
        self.last_track_id = track_id
        self.detection_count += 1
        self.cameras_seen.add(camera_id)
        if crop is not None:
            self.best_body_crop = crop
    
    def seconds_since_seen(self) -> float:
        return time.time() - self.last_seen


class GlobalPersonPool:
    """
    Barcha AIWorker lar uchun umumiy person pool.
    
    Xususiyatlari:
    - Thread-safe (bir nechta kameralar parallel)
    - Face matching (cosine similarity)
    - Body matching (histogram correlation)
    - Temporal logic (yaqin vaqt = yaqin masofa)
    - TTL cleanup (120 sek ko'rinmasa → o'chirish)
    """
    
    # Thresholds
    FACE_MATCH_THRESH = 0.45   # Yuzdan tanish
    BODY_MATCH_THRESH = 0.65   # Body dan tanish (qattiqroq)
    TEMPORAL_BONUS_SEC = 30    # 30 sek ichida ko'rilgan bo'lsa → bonus
    TEMPORAL_BONUS = 0.15      # +0.15 similarity
    TTL_SECONDS = 120          # 120 sek ko'rinmasa → o'chirish
    
    def __init__(self):
        self.lock = threading.Lock()
        
        # person_id → Person object
        self.persons: Dict[int, Person] = {}
        
        # (camera_id, track_id) → person_id mapping
        self.track_to_person: Dict[Tuple[str, int], int] = {}
        
        # Next ID
        self._next_person_id = 1
        
        # Body ReID engine
        from backend.ai.reid_engine import BodyReIDEngine
        self.body_engine = BodyReIDEngine()
        
        print(f"[GlobalPersonPool] ✅ Initialized (face={self.FACE_MATCH_THRESH}, body={self.BODY_MATCH_THRESH})", flush=True)
    
    def _get_next_id(self) -> int:
        """Keyingi unikal ID"""
        with self.lock:
            pid = self._next_person_id
            self._next_person_id += 1
            return pid
    
    def _cosine_sim(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity"""
        if emb1 is None or emb2 is None:
            return 0.0
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (n1 * n2))
    
    def match_or_create(
        self,
        camera_id: str,
        track_id: int,
        face_emb: Optional[np.ndarray] = None,
        face_conf: float = 0.0,
        body_crop: Optional[np.ndarray] = None,
        known_name: Optional[str] = None,
    ) -> Tuple[int, str, str]:
        """
        Asosiy metod: track uchun person match qilish yoki yaratish.
        
        Returns: (person_id, name, method)
            method: "face", "body", "track_cache", yoki "new"
        """
        with self.lock:
            # 1) TRACK CACHE: bu track allaqachon ma'lummi?
            track_key = (camera_id, track_id)
            if track_key in self.track_to_person:
                pid = self.track_to_person[track_key]
                if pid in self.persons:
                    person = self.persons[pid]
                    # Yangilash
                    if face_emb is not None and face_conf >= self.FACE_MATCH_THRESH:
                        person.update_from_face(face_emb, face_conf, camera_id, track_id)
                    elif body_crop is not None:
                        body_hist = self.body_engine.extract_features(body_crop)
                        if body_hist is not None:
                            person.update_from_body(body_hist, camera_id, track_id, body_crop)
                    else:
                        person.last_seen = time.time()
                        person.last_camera = camera_id
                    
                    return pid, person.name, "track_cache"
            
            # 2) FACE MATCH: face embedding bilan qidirish
            if face_emb is not None and face_conf >= self.FACE_MATCH_THRESH:
                best_pid = None
                best_sim = 0.0
                best_person = None
                
                for pid, person in self.persons.items():
                    if person.face_emb is None:
                        continue
                    
                    sim = self._cosine_sim(face_emb, person.face_emb)
                    
                    # Temporal bonus
                    if person.seconds_since_seen() < self.TEMPORAL_BONUS_SEC:
                        sim += self.TEMPORAL_BONUS
                    
                    if sim > best_sim:
                        best_sim = sim
                        best_pid = pid
                        best_person = person
                
                if best_sim >= self.FACE_MATCH_THRESH and best_person is not None:
                    best_person.update_from_face(face_emb, face_conf, camera_id, track_id)
                    self.track_to_person[track_key] = best_pid
                    return best_pid, best_person.name, "face"
            
            # 3) BODY MATCH: body histogram bilan qidirish
            if body_crop is not None:
                body_hist = self.body_engine.extract_features(body_crop)
                if body_hist is not None:
                    best_pid = None
                    best_sim = 0.0
                    best_person = None
                    
                    for pid, person in self.persons.items():
                        if person.body_hist is None:
                            continue
                        
                        sim = self.body_engine.compute_similarity(body_hist, person.body_hist)
                        
                        # Temporal bonus
                        if person.seconds_since_seen() < self.TEMPORAL_BONUS_SEC:
                            sim += self.TEMPORAL_BONUS
                        
                        if sim > best_sim:
                            best_sim = sim
                            best_pid = pid
                            best_person = person
                    
                    if best_sim >= self.BODY_MATCH_THRESH and best_person is not None:
                        best_person.update_from_body(body_hist, camera_id, track_id, body_crop)
                        self.track_to_person[track_key] = best_pid
                        return best_pid, best_person.name, "body"
            
            # 4) YANGI PERSON yaratish
            new_pid = self._next_person_id
            self._next_person_id += 1
            
            if known_name:
                name = known_name
                is_known = True
            else:
                name = f"Person-{new_pid}"
                is_known = False
            
            person = Person(new_pid, name, is_known)
            
            if face_emb is not None:
                person.update_from_face(face_emb, face_conf, camera_id, track_id)
            if body_crop is not None:
                body_hist = self.body_engine.extract_features(body_crop)
                if body_hist is not None:
                    person.update_from_body(body_hist, camera_id, track_id, body_crop)
            
            self.persons[new_pid] = person
            self.track_to_person[track_key] = new_pid
            
            return new_pid, name, "new"
    
    def cleanup_old(self):
        """Eski (TTL dan oshgan) person larni o'chirish"""
        now = time.time()
        with self.lock:
            to_remove = []
            for pid, person in self.persons.items():
                # Faqat unknown va kam ko'rilganlarni o'chirish
                if (not person.is_known and 
                    person.detection_count < 3 and 
                    now - person.last_seen > self.TTL_SECONDS):
                    to_remove.append(pid)
            
            for pid in to_remove:
                del self.persons[pid]
                # Track mapping ni ham tozalash
                keys_to_del = [k for k, v in self.track_to_person.items() if v == pid]
                for k in keys_to_del:
                    del self.track_to_person[k]
            
            if to_remove:
                print(f"[GlobalPersonPool] 🧹 Cleaned up {len(to_remove)} old persons", flush=True)
    
    def get_stats(self) -> dict:
        with self.lock:
            known = sum(1 for p in self.persons.values() if p.is_known)
            unknown = len(self.persons) - known
            return {
                "total": len(self.persons),
                "known": known,
                "unknown": unknown,
                "active_tracks": len(self.track_to_person),
            }
    
    def get_all_persons(self):
        """UI uchun barcha person larni qaytarish"""
        with self.lock:
            return list(self.persons.values())
