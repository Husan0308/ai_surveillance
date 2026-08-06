import threading
import time
from PySide6.QtCore import QThread
from backend.core.logger import get_logger

log = get_logger("core.batch_scheduler")


class BatchScheduler(QThread):
    """
    Markaziy GPU batch scheduler.
    
    Barcha kameralardan frame yig'ib, bitta GPU call qiladi.
    3-5x tezroq inference, past latency.
    """
    
    def __init__(self, detector, batch_size=6, interval_ms=40):
        super().__init__()
        self.detector = detector
        self.batch_size = batch_size
        self.interval = interval_ms / 1000.0
        
        self._cond = threading.Condition()
        self._pending_frames = {}  # camera_id -> (frame, callback, frame_id)
        self._running = False
        
        log.info(
            "BatchScheduler initialized: batch_size=%d, interval=%.1fms",
            batch_size, interval_ms
        )
    
    def submit(self, camera_id, frame, frame_id, callback):
        """AIWorker dan frame qabul qilish."""
        with self._cond:
            self._pending_frames[camera_id] = (frame, callback, frame_id)
            self._cond.notify_all()
    
    def run(self):
        """Asosiy loop - 6 ta kadrni yig'ib, bitta GPU chaqiruvi bilan ishlash."""
        self._running = True
        log.info("BatchScheduler started (6-batch 1 GPU call mode)")
        
        while self._running:
            with self._cond:
                # Frame kelishini kutish (25ms takrorlanish)
                if not self._pending_frames:
                    self._cond.wait(timeout=0.025)

                if not self._pending_frames:
                    continue

                # Barcha 6 ta kamerani 1 ta GPU call ga yig'ish (maksimum 45ms kutish)
                t_end = time.time() + 0.045
                while len(self._pending_frames) < self.batch_size and self._running:
                    rem = t_end - time.time()
                    if rem <= 0:
                        break
                    self._cond.wait(timeout=rem)

                if not self._pending_frames:
                    continue

                camera_ids = list(self._pending_frames.keys())
                frames = [self._pending_frames[cid][0] for cid in camera_ids]
                callbacks = {cid: self._pending_frames[cid][1] for cid in camera_ids}
                self._pending_frames.clear()
            
            # GPU batch inference (1 ta chaqiruv barcha kameralar uchun!)
            try:
                t0 = time.time()
                all_detections = self.detector.detect_batch(frames)
                infer_ms = (time.time() - t0) * 1000
                
                print(f"[BatchScheduler] ⚡ MEGA BATCH GPU CALL: {len(frames)} kameralar bitta GPU call bilan ishlandi ({infer_ms:.1f}ms)", flush=True)
            except Exception as e:
                log.error("Batch inference error: %s", e)
                all_detections = [[] for _ in frames]
            
            # Natijalarni har bir AIWorker ga bir vaqtda qaytarish
            for camera_id, detections in zip(camera_ids, all_detections):
                try:
                    callbacks[camera_id](detections)
                except Exception as e:
                    log.error("Callback error for %s: %s", camera_id, e)
        
        log.info("BatchScheduler stopped")
    
    def stop(self):
        """Thread ni to'xtatish."""
        self._running = False
        with self._cond:
            self._cond.notify_all()
