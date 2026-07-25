import time
import threading


class FrameBuffer:
    """
    Har kamera uchun eng oxirgi frame saqlanadi.
    UI va AI worker bu yerdan oladi.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.timestamp = 0.0
        self.frame_id = 0

    def put(self, frame):
        with self.lock:
            self.frame = frame
            self.timestamp = time.time()
            self.frame_id += 1

    def get(self):
        with self.lock:
            return self.frame, self.timestamp, self.frame_id

    def clear(self):
        with self.lock:
            self.frame = None
            self.timestamp = 0.0
            self.frame_id = 0