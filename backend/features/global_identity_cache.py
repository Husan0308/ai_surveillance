"""
Global Identity Cache.
Face embedding + Person crop histogram.
Cross-camera identity tracking.
"""
import time
import cv2
import numpy as np
import threading


class GlobalIdentityCache:
    def __init__(self, face_threshold=0.45, crop_threshold=0.70, ttl=300.0):
        self.entries = []  # [(person_id, name, face_emb, crop_hist, last_seen)]
        self.face_threshold = face_threshold
        self.crop_threshold = crop_threshold
        self.ttl = ttl
        self.lock = threading.Lock()

    @staticmethod
    def crop_histogram(crop_bgr):
        """Person crop dan color histogram (kiyim tanish uchun)"""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        crop = cv2.resize(crop_bgr, (64, 128))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 16],
                           [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.astype(np.float32)

    def add_or_update(self, person_id, name, face_emb=None, crop_hist=None):
        """Identity ni cache ga qo'shish/yangilash"""
        now = time.time()
        with self.lock:
            for i, (pid, nm, fe, ch, ls) in enumerate(self.entries):
                if pid == person_id:
                    self.entries[i] = (
                        pid, nm,
                        face_emb if face_emb is not None else fe,
                        crop_hist if crop_hist is not None else ch,
                        now
                    )
                    return
            self.entries.append((person_id, name, face_emb, crop_hist, now))

    def match_by_crop(self, crop_hist):
        """Crop histogram bilan solishtirish (yuz ko'rinmaganda)"""
        if crop_hist is None:
            return None, None, 0.0
        now = time.time()
        with self.lock:
            self.entries = [e for e in self.entries if now - e[4] < self.ttl]
            best_pid, best_name, best_sim = None, None, 0.0
            for pid, nm, fe, ch, ls in self.entries:
                if ch is None:
                    continue
                sim = float(cv2.compareHist(
                    crop_hist.reshape(-1, 1).astype(np.float32),
                    ch.reshape(-1, 1).astype(np.float32),
                    cv2.HISTCMP_CORREL
                ))
                if sim > best_sim:
                    best_sim, best_pid, best_name = sim, pid, nm
            if best_sim >= self.crop_threshold:
                return best_pid, best_name, best_sim
        return None, None, best_sim

    def match_by_face(self, face_emb):
        """Face embedding bilan solishtirish"""
        if face_emb is None:
            return None, None, 0.0
        now = time.time()
        with self.lock:
            self.entries = [e for e in self.entries if now - e[4] < self.ttl]
            best_pid, best_name, best_sim = None, None, 0.0
            for pid, nm, fe, ch, ls in self.entries:
                if fe is None:
                    continue
                sim = float(np.dot(face_emb, fe))
                if sim > best_sim:
                    best_sim, best_pid, best_name = sim, pid, nm
            if best_sim >= self.face_threshold:
                return best_pid, best_name, best_sim
        return None, None, best_sim
