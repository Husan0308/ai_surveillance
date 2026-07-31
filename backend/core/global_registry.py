"""
Global Identity Registry — YAGONA MARKAZIY ID GENERATOR
Barcha kameralar uchun umumiy. Hech qachon bir xil ID ikki odamga berilmaydi.
Thread-safe singleton pattern.
"""
import threading
import time
from typing import Optional, Dict, Tuple

class GlobalIdentityRegistry:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern — barcha kameralar bitta instance ishlatadi"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._data_lock = threading.Lock()
        
        # === ASOSIY MA'LUMOTLAR ===
        # person_id → {name, face_emb, body_hist, last_seen, cameras}
        self.persons: Dict[int, dict] = {}
        
        # (camera_id, track_id) → person_id
        self.track_map: Dict[Tuple[str, int], int] = {}
        
        # Keyingi available ID (hech qachon orqaga qaytmaydi)
        self._next_id: int = 1
        
        # DB dan yuklangan max ID (restartdan keyin davom ettirish)
        self._db_max_id: int = 0
        
        self._initialized = True
        print(f"[GlobalRegistry] ✅ Singleton initialized", flush=True)
    
    def load_from_db(self, db):
        """DB dan eng katta ID ni yuklash (restartdan keyin takrorlanmasligi uchun)"""
        try:
            with db.lock:
                # persons jadvalidan max id
                row = db.conn.execute("SELECT MAX(id) as max_id FROM persons").fetchone()
                db_max = row["max_id"] if row and row["max_id"] else 0
                
                # events jadvalidan UNK-* nomlaridan max id
                rows = db.conn.execute("""
                    SELECT person_name FROM events 
                    WHERE person_name LIKE 'UNK-%' OR person_name LIKE 'Person-%'
                    ORDER BY id DESC LIMIT 200
                """).fetchall()
                
                unk_max = 0
                for r in rows:
                    name = r["person_name"] if isinstance(r, dict) else r[0]
                    try:
                        num = int(name.split("-")[-1])
                        if num > unk_max:
                            unk_max = num
                    except:
                        pass
                
                self._db_max_id = max(db_max, unk_max)
                self._next_id = self._db_max_id + 1
                
                print(f"[GlobalRegistry] 📦 DB max_id={self._db_max_id}, next_id={self._next_id}", flush=True)
        except Exception as e:
            print(f"[GlobalRegistry] ⚠ load_from_db error: {e}", flush=True)
    
    def _allocate_id(self) -> int:
        """Yangi unikal ID ajratish (thread-safe, hech qachon takrorlanmaydi)"""
        with self._data_lock:
            new_id = self._next_id
            self._next_id += 1
            return new_id
    
    def get_or_create_person(
        self,
        camera_id: str,
        track_id: int,
        face_emb=None,
        body_hist=None,
        known_name: Optional[str] = None,
        known_db_id: Optional[int] = None,
    ) -> Tuple[int, str, bool]:
        """
        Asosiy metod: track uchun shaxs olish yoki yaratish.
        
        Returns: (person_id, name, is_new)
        """
        with self._data_lock:
            track_key = (camera_id, track_id)
            
            # 1) Track allaqachon ma'lummi?
            if track_key in self.track_map:
                pid = self.track_map[track_key]
                if pid in self.persons:
                    p = self.persons[pid]
                    p["last_seen"] = time.time()
                    p["cameras"].add(camera_id)
                    if face_emb is not None:
                        p["face_emb"] = face_emb
                    if body_hist is not None:
                        p["body_hist"] = body_hist
                    return pid, p["name"], False
            
            # 2) Known person (DB da bor) → DB ID ishlatish
            if known_db_id is not None:
                pid = known_db_id
                name = known_name or f"Person-{pid}"
                
                if pid not in self.persons:
                    self.persons[pid] = {
                        "name": name,
                        "face_emb": face_emb,
                        "body_hist": body_hist,
                        "last_seen": time.time(),
                        "cameras": {camera_id},
                        "is_known": True,
                    }
                else:
                    self.persons[pid]["last_seen"] = time.time()
                    self.persons[pid]["cameras"].add(camera_id)
                    if face_emb is not None:
                        self.persons[pid]["face_emb"] = face_emb
                    if body_hist is not None:
                        self.persons[pid]["body_hist"] = body_hist
                
                self.track_map[track_key] = pid
                return pid, name, False
            
            # 3) Unknown person → YANGI GLOBAL ID
            new_id = self._allocate_id()
            name = f"UNK-{new_id}"
            
            self.persons[new_id] = {
                "name": name,
                "face_emb": face_emb,
                "body_hist": body_hist,
                "last_seen": time.time(),
                "cameras": {camera_id},
                "is_known": False,
            }
            
            self.track_map[track_key] = new_id
            
            print(f"[GlobalRegistry] 🆕 {name} (id={new_id}) cam={camera_id} trk={track_id}", flush=True)
            return new_id, name, True
    
    def bind_track(self, camera_id: str, track_id: int, person_id: int):
        """Trackni mavjud person ga bog'lash (Re-ID match bo'lganda)"""
        with self._data_lock:
            track_key = (camera_id, track_id)
            self.track_map[track_key] = person_id
            if person_id in self.persons:
                self.persons[person_id]["last_seen"] = time.time()
                self.persons[person_id]["cameras"].add(camera_id)
    
    def get_person_id(self, camera_id: str, track_id: int) -> Optional[int]:
        """Track uchun person_id olish"""
        with self._data_lock:
            return self.track_map.get((camera_id, track_id))
    
    def cleanup_old_tracks(self, ttl: float = 120.0):
        """Eski track larni tozalash"""
        now = time.time()
        with self._data_lock:
            to_remove = []
            for key, pid in list(self.track_map.items()):
                if pid in self.persons:
                    if now - self.persons[pid]["last_seen"] > ttl:
                        to_remove.append(key)
            
            for key in to_remove:
                del self.track_map[key]
            
            if to_remove:
                print(f"[GlobalRegistry] 🧹 Cleaned {len(to_remove)} old tracks", flush=True)
    
    def get_stats(self) -> dict:
        with self._data_lock:
            known = sum(1 for p in self.persons.values() if p.get("is_known"))
            return {
                "total_persons": len(self.persons),
                "known": known,
                "unknown": len(self.persons) - known,
                "active_tracks": len(self.track_map),
                "next_id": self._next_id,
            }
