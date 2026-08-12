from __future__ import annotations
import time
from .camera_worker import CameraWorker
from .latest_frame import LatestFrameStore

class CameraManager:
    def __init__(self,cameras,core_config=None):
        self.core_config=dict(core_config or {})
        enabled=[c for c in cameras if c.get("online",True)]
        self.stores={str(c["id"]):LatestFrameStore() for c in enabled}
        self.workers={str(c["id"]):CameraWorker(c,self.stores[str(c["id"])],self.core_config) for c in enabled}
    def start(self):
        # Avoid six RTSP/NVDEC pipelines negotiating at exactly the same instant.
        stagger=max(0.0,float(self.core_config.get("startup_stagger_sec",0.20)))
        for index,worker in enumerate(self.workers.values()):
            worker.start()
            if stagger and index+1<len(self.workers):time.sleep(stagger)
    def stop(self):
        for worker in self.workers.values():worker.stop()
        for worker in self.workers.values():worker.join(6)
    def metrics(self):return {cid:worker.metrics() for cid,worker in self.workers.items()}
