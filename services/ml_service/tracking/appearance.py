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


def reid_crop_quality(crop,confidence=0.0):
    """Cheap CPU quality gate for asynchronous body-ReID evidence."""
    import cv2
    if crop is None or not getattr(crop,"size",0):return {"score":0.0,"width":0,"height":0,"blur_variance":0.0,"reason":"empty_crop"}
    height,width=crop.shape[:2];aspect=width/max(height,1);gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);blur=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    size_score=min(1.0,min(width/60.0,height/160.0));sharp_score=min(1.0,blur/120.0);aspect_score=1.0 if .12<=aspect<=1.2 else 0.0;score=.45*size_score+.30*sharp_score+.15*aspect_score+.10*min(1.0,max(0.0,float(confidence)))
    reason="quality_ok" if width>=20 and height>=45 and aspect_score and blur>=12 else ("crop_too_small" if width<20 or height<45 else "invalid_person_crop_aspect" if not aspect_score else "crop_too_blurry")
    return {"score":float(score),"width":int(width),"height":int(height),"blur_variance":blur,"reason":reason}
