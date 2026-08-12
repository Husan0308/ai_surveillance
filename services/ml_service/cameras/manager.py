from __future__ import annotations
import threading
from shared.logging import get_logger
from .buffer import LatestFrameBuffer
from .reader import CameraReader

log=get_logger(__name__)

class CameraManager:
    """Thread-safe owner enforcing one active inference reader per camera."""
    @staticmethod
    def _reader_config(config):
        result=dict(config);result["source"]=result.get("ai_source") or result.get("source");result.pop("display_source",None);result.pop("display_codec",None);result.pop("recovery_rois",None);return result
    def __init__(self, on_frame_available=None, capture_factory=None, on_display_frame=None):
        self._lock = threading.RLock()
        self._readers, self._buffers, self._configs = {}, {}, {}
        self._on_available, self._factory, self._on_display_frame = on_frame_available, capture_factory, on_display_frame;self._running=False

    def configure(self, configs):
        desired = {str(c["id"]): self._reader_config(c) for c in configs if c.get("online", c.get("enabled", False))}
        with self._lock: remove = set(self._readers) - set(desired)
        for camera_id in remove: self.remove(camera_id)
        for camera_id, config in desired.items():
            with self._lock: exists = camera_id in self._readers;changed = exists and self._configs.get(camera_id) != dict(config)
            if changed:
                self.remove(camera_id);exists=False
            if not exists:
                buffer = LatestFrameBuffer(self._on_available)
                reader = CameraReader(config, buffer, self._factory, self._on_display_frame)
                with self._lock:
                    self._buffers[camera_id], self._readers[camera_id], self._configs[camera_id] = buffer, reader, dict(config)
                if self._running:reader.start()

    def start(self):
        self._running=True
        with self._lock: readers = list(self._readers.values())
        for reader in readers: reader.start()

    def remove(self, camera_id):
        with self._lock:
            reader = self._readers.pop(camera_id, None); buffer = self._buffers.pop(camera_id, None);self._configs.pop(camera_id,None)
        if reader:
            reader.stop()
            if not reader.join(6):log.error("Camera worker did not stop: %s",camera_id)
        if buffer: buffer.close()

    def shutdown(self, join_timeout=6.0):
        self._running=False
        with self._lock:
            readers, buffers = list(self._readers.values()), list(self._buffers.values())
        for reader in readers: reader.stop()
        for buffer in buffers: buffer.close()
        for reader in readers:
            if not reader.join(join_timeout):log.error("Camera worker did not stop: %s",reader.camera_id)

    def buffers(self):
        with self._lock: return dict(self._buffers)

    def metrics(self):
        with self._lock: readers = dict(self._readers)
        return {cid: reader.metrics() for cid, reader in readers.items()}

    def reader_count(self):
        with self._lock:return len(self._readers)

    def reader_ids(self):
        with self._lock:return tuple(sorted(self._readers))
