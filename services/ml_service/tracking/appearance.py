"""Optional, globally batched appearance enrichment interface."""
import time
import numpy as np

class AppearanceExtractor:
    def __init__(self, model=None, device="cpu", batch_size=32):
        self.model, self.device, self.batch_size = model, device, max(1, int(batch_size))

    @property
    def available(self): return self.model is not None

    def extract_batch(self, crops):
        """One logical batch API; model implementations may chunk at configured capacity."""
        if not crops or not self.available: return [None] * len(crops), {"preprocess_ms": 0.0, "gpu_ms": 0.0}
        started = time.perf_counter(); outputs = []
        preprocess_ms = 0.0; gpu_ms = 0.0
        for offset in range(0, len(crops), self.batch_size):
            chunk = crops[offset:offset + self.batch_size]
            before = time.perf_counter()
            result = self.model.extract_batch(chunk)
            elapsed = (time.perf_counter() - before) * 1000
            if isinstance(result, tuple): embeddings, timing = result; gpu_ms += float(timing.get("gpu_ms", elapsed))
            else: embeddings, gpu_ms = result, gpu_ms + elapsed
            outputs.extend(embeddings)
        normalized = []
        for embedding in outputs:
            if embedding is None: normalized.append(None); continue
            value = np.asarray(embedding, np.float32); norm = np.linalg.norm(value)
            normalized.append(value / norm if norm else value)
        return normalized, {"preprocess_ms": preprocess_ms, "gpu_ms": gpu_ms,
                            "total_ms": (time.perf_counter() - started) * 1000}

def crop_detection(frame, bbox):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, x2 = max(0, x1), min(width, x2); y1, y2 = max(0, y1), min(height, y2)
    return frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
