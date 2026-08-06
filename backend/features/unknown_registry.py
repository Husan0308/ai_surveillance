"""
Global Unknown Identity Registry.
Bir odam barcha kamerada bitta UNK-ID oladi.
Embedding cosine similarity bo'yicha solishtiriladi.
"""
import time
import numpy as np
import threading


class UnknownRegistry:
    def __init__(self, match_threshold=0.45, ttl=300.0):
        self.entries = []  # [(unk_id, embedding, first_seen, last_seen)]
        self.next_id = 1
        self.threshold = match_threshold
        self.ttl = ttl  # 5 min ko'rinmasa → o'chirish
        self.lock = threading.Lock()

    def match_or_create(self, embedding):
        """
        Embedding ni mavjud unknown lar bilan solishtirish.
        
        Returns: (unk_id: str, is_new: bool)
        """
        if embedding is None:
            with self.lock:
                uid = f"UNK-{self.next_id}"
                self.next_id += 1
            return uid, True

        emb = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        now = time.time()

        with self.lock:
            # Eski entry larni tozalash
            self.entries = [
                e for e in self.entries
                if now - e[3] < self.ttl
            ]

            # Mavjud unknown lar bilan solishtirish
            best_id = None
            best_sim = 0.0
            best_idx = -1

            for i, (uid, gal_emb, first, last) in enumerate(self.entries):
                sim = float(np.dot(emb, gal_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_id = uid
                    best_idx = i

            if best_id is not None and best_sim >= self.threshold:
                # Mavjud unknown → last_seen yangilash
                self.entries[best_idx] = (
                    best_id,
                    self.entries[best_idx][1],
                    self.entries[best_idx][2],
                    now
                )
                return best_id, False

            # Yangi unknown
            uid = f"UNK-{self.next_id}"
            self.next_id += 1
            self.entries.append((uid, emb, now, now))
            return uid, True

    def get_all(self):
        with self.lock:
            return [(e[0], e[2], e[3]) for e in self.entries]

    def count(self):
        with self.lock:
            return len(self.entries)
