"""
Room Manager: Kamera → Xona mapping + xona ichida ReID + pozitsiya tracking.
"""
import time
import threading


# Kamera → Xona mapping
ROOM_MAP = {
    "CAM-01": "Room-A", "CAM-06": "Room-A",
    "CAM-02": "Room-B", "CAM-05": "Room-B",
    "CAM-03": "Room-C", "CAM-04": "Room-C",
}


class RoomManager:
    def __init__(self):
        self.lock = threading.Lock()
        # person_id → pozitsiya ma'lumoti
        self.positions = {}
        # xona → {person_id: (crop_hist, face_emb, last_seen, camera_id)}
        self.room_cache = {}

    def get_room(self, camera_id):
        return ROOM_MAP.get(camera_id, f"Unknown-{camera_id}")

    def get_cameras_in_room(self, room_id):
        return [cam for cam, room in ROOM_MAP.items() if room == room_id]

    def update_position(self, person_id, name, camera_id, bbox, frame_w, frame_h):
        """Odam pozitsiyasini yangilash"""
        room = self.get_room(camera_id)
        x1, y1, x2, y2 = bbox[:4]
        cx = ((x1 + x2) / 2.0) / max(1, frame_w)
        cy = ((y1 + y2) / 2.0) / max(1, frame_h)

        h_pos = "chap" if cx < 0.33 else "o'ng" if cx > 0.66 else "markaz"
        v_pos = "yuqori" if cy < 0.33 else "past" if cy > 0.66 else "o'rta"

        with self.lock:
            self.positions[person_id] = {
                "name": name,
                "room": room,
                "camera_id": camera_id,
                "x": round(cx, 2),
                "y": round(cy, 2),
                "position_text": f"{room} · {camera_id} · {v_pos}-{h_pos}",
                "time": time.time(),
            }

    def get_position(self, person_id):
        with self.lock:
            pos = self.positions.get(person_id)
            if pos and time.time() - pos["time"] < 300:
                return pos
        return None

    def get_position_text(self, person_id):
        pos = self.get_position(person_id)
        return pos["position_text"] if pos else "Noma'lum"

    def save_to_room(self, person_id, name, camera_id, face_emb=None, crop_hist=None):
        """Odamni xona cache ga saqlash (ReID uchun)"""
        room = self.get_room(camera_id)
        with self.lock:
            if room not in self.room_cache:
                self.room_cache[room] = {}
            existing = self.room_cache[room].get(person_id, {})
            self.room_cache[room][person_id] = {
                "name": name,
                "face_emb": face_emb if face_emb is not None else existing.get("face_emb"),
                "crop_hist": crop_hist if crop_hist is not None else existing.get("crop_hist"),
                "camera_id": camera_id,
                "time": time.time(),
            }

    def match_in_room(self, camera_id, crop_hist=None, face_emb=None, threshold=0.65):
        """Xona ichida odamni tanish (ReID)"""
        import numpy as np
        import cv2
        room = self.get_room(camera_id)
        now = time.time()

        with self.lock:
            entries = self.room_cache.get(room, {})
            best_pid, best_name, best_sim = None, None, 0.0

            for pid, info in entries.items():
                if now - info["time"] > 300:
                    continue

                sim = 0.0
                # Face embedding bilan solishtirish (aniqroq)
                if face_emb is not None and info.get("face_emb") is not None:
                    sim = float(np.dot(face_emb, info["face_emb"]))
                # Crop histogram bilan (kiyim)
                elif crop_hist is not None and info.get("crop_hist") is not None:
                    sim = float(cv2.compareHist(
                        crop_hist.reshape(-1, 1).astype(np.float32),
                        info["crop_hist"].reshape(-1, 1).astype(np.float32),
                        cv2.HISTCMP_CORREL
                    ))

                if sim > best_sim:
                    best_sim, best_pid, best_name = sim, pid, info["name"]

            if best_sim >= threshold:
                return best_pid, best_name, best_sim

        return None, None, 0.0
