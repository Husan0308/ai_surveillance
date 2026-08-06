"""
SharedReIDGallery — thread-safe, CROSS-CAMERA shared ReID gallery.

Bitta instance ServiceManager da → hamma AIWorker lar ulashadi.
Natija: CAM-02 da tanilgan odamni CAM-06 DARHOL taniydi.
"""
import threading
import time
import numpy as np


class SharedReIDGallery:
    def __init__(self, max_per_person=5):
        self._lock = threading.RLock()
        self.features = {}        # person_id → [feat array]
        self.names = {}           # person_id → name
        self.last_camera = {}     # person_id → camera_id
        self.last_room = {}       # person_id → room/location
        self.last_seen = {}       # person_id → timestamp
        self.max_per_person = max_per_person
        self.reservations = {}

    def add(self, person_id, name, feature, camera_id=None, room=None):
        if feature is None or person_id is None:
            return
        with self._lock:
            gal = self.features.setdefault(person_id, [])
            gal.append(feature)
            if len(gal) > self.max_per_person:
                gal.pop(0)
            if name:
                self.names[person_id] = name
            if camera_id:
                self.last_camera[person_id] = camera_id
            if room:
                self.last_room[person_id] = room
            self.last_seen[person_id] = time.time()

    def match(self, feature, threshold, camera_id=None, track_id=None):
        """Eng yaxshi match ni qaytaradi: (person_id, name, score)."""
        if feature is None:
            return None, None, 0.0
        best_pid, best_score = None, 0.0
        with self._lock:
            now = time.time()
            for pid, feats in self.features.items():
                held = self.reservations.get((pid, camera_id))
                if held and held[0] == camera_id and held[1] != track_id and now - held[2] < 3.0:
                    continue
                for gf in feats:
                    sim = float(np.dot(feature, gf))
                    if sim > best_score:
                        best_score = sim
                        best_pid = pid
            name = self.names.get(best_pid) if best_pid is not None else None
            if best_pid is not None and best_score >= threshold and camera_id is not None:
                self.reservations[(best_pid, camera_id)] = (camera_id, track_id, now)
            for pid, held in list(self.reservations.items()):
                if now - held[2] > 15.0:
                    self.reservations.pop(pid, None)
        if best_score >= threshold:
            return best_pid, name, best_score
        return None, None, best_score

    def get_location(self, person_id):
        with self._lock:
            return (self.last_camera.get(person_id),
                    self.last_room.get(person_id))

    def update_location(self, person_id, camera_id, room):
        with self._lock:
            if camera_id:
                self.last_camera[person_id] = camera_id
            if room:
                self.last_room[person_id] = room
            self.last_seen[person_id] = time.time()

    def size(self):
        with self._lock:
            return len(self.features)
