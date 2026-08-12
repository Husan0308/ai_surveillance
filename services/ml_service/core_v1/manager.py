from __future__ import annotations
from .camera_worker import CameraWorker
from .latest_frame import LatestFrameStore

class CameraManager:
    def __init__(self,cameras):
        enabled=[c for c in cameras if c.get("online",True)]
        self.stores={str(c["id"]):LatestFrameStore() for c in enabled}
        self.workers={str(c["id"]):CameraWorker(c,self.stores[str(c["id"])]) for c in enabled}
    def start(self):
        for worker in self.workers.values():worker.start()
    def stop(self):
        for worker in self.workers.values():worker.stop()
        for worker in self.workers.values():worker.join(6)
    def metrics(self):return {cid:worker.metrics() for cid,worker in self.workers.items()}
