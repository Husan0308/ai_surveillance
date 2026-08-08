"""Camera/batching diagnostic synthetic capture support."""
import time

class _Frame: shape = (360, 640, 3)

class SyntheticCapture:
    def __init__(self, config):
        self.camera_id, self.opened = config["id"], True
        self.interval = 1.0 / float(config.get("fps", 20)); self.next_frame = time.monotonic(); self.frames = 0
    def isOpened(self): return self.opened
    def read(self):
        delay = self.next_frame - time.monotonic()
        if delay > 0: time.sleep(delay)
        self.next_frame = time.monotonic() + self.interval; self.frames += 1
        if self.camera_id == "CAM-02" and self.frames > 3:
            time.sleep(.55); return False, None
        return self.opened, _Frame()
    def release(self): self.opened = False

def synthetic_capture_factory(config): return SyntheticCapture(config)
