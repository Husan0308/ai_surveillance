"""Correctly named detector timing and throughput metrics."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import threading
import time

@dataclass(frozen=True)
class DetectorBatchMetrics:
    batch_id: int
    batch_size: int
    preprocess_ms: float
    cpu_pack_ms: float
    h2d_ms: float
    gpu_inference_ms: float
    postprocess_ms: float
    result_parse_ms: float
    detector_wall_ms: float
    total_detection_latency_ms: float

class DetectorMetrics:
    def __init__(self):
        self._lock = threading.Lock(); self._started = time.monotonic()
        self._batches = self.processed_frames_total = 0
        self.stale_drops_before_inference = self.duplicate_inference_prevented = 0
        self.last: DetectorBatchMetrics | None = None

    def record(self, item):
        with self._lock:
            self.last = item; self._batches += 1; self.processed_frames_total += item.batch_size

    def snapshot(self):
        with self._lock:
            elapsed = max(1e-6, time.monotonic() - self._started)
            result = asdict(self.last) if self.last else {}
            result.update(batch_rate=self._batches / elapsed,
                          processed_frames_total=self.processed_frames_total,
                          stale_drops_before_inference=self.stale_drops_before_inference,
                          duplicate_inference_prevented=self.duplicate_inference_prevented)
            return result

    def format_compact(self):
        item = self.snapshot()
        if not self.last: return "Detector waiting for first batch"
        return (f"GPU batch: {item['batch_size']}\npreprocess: {item['preprocess_ms']:.1f} ms "
                f"cpu_pack: {item['cpu_pack_ms']:.1f} ms h2d: {item['h2d_ms']:.1f} ms\n"
                f"gpu inference: {item['gpu_inference_ms']:.1f} ms postprocess: {item['postprocess_ms']:.1f} ms\n"
                f"detector wall: {item['detector_wall_ms']:.1f} ms rate: {item['batch_rate']:.1f} batches/s")
