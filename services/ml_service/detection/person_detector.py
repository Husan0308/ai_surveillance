"""Permanent, dynamically batched person-only YOLO detector."""
from collections import deque
import threading
import time
import numpy as np
from services.ml_service.detection.schemas import Detection, CameraDetectionResult, DetectionBatchResult
from services.ml_service.metrics.detector_metrics import DetectorBatchMetrics, DetectorMetrics
from services.ml_service.pipeline.batch import BatchOutput
from services.ml_service.pipeline.preprocessing import BatchPreprocessor, original_bbox
from shared.logging import get_logger
log = get_logger(__name__)

class UltralyticsBackend:
    coordinates_original = True
    def __init__(self, config, max_batch_size=6):
        import torch
        from ultralytics import YOLO
        cfg = config.get("ai", {}).get("detector", {})
        self.torch = torch; requested = cfg.get("device", "auto")
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if requested == "auto" else str(requested)
        self.conf, self.iou = float(cfg.get("conf", .3)), float(cfg.get("iou", .45))
        self.max_det, self.half = int(cfg.get("max_det", 50)), bool(cfg.get("half", False))
        raw_size = cfg.get("imgsz", (448, 800)); self.imgsz = tuple(raw_size) if isinstance(raw_size, (list, tuple)) else (raw_size, raw_size)
        self.model = YOLO(cfg.get("model", "models/yolo26n.pt")); self.model.to(self.device)
        dummy = np.zeros((self.imgsz[0], self.imgsz[1], 3), dtype=np.uint8)
        with torch.inference_mode():
            self.model.predict([dummy] * max(1, int(max_batch_size)), classes=[0], conf=self.conf,
                iou=self.iou, max_det=self.max_det, imgsz=self.imgsz, device=self.device,
                half=self.half, verbose=False)
        log.info("PersonDetector warmup complete: device=%s max_batch=%d", self.device, max_batch_size)

    def infer(self, prepared):
        torch, predictor = self.torch, self.model.predictor
        cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")
        dtype = torch.float16 if getattr(predictor.model, "fp16", False) else torch.float32
        with torch.inference_mode():
            host = torch.from_numpy(prepared.images_nchw)
            if cuda:
                host = host.pin_memory(); stream = torch.cuda.current_stream(self.device)
                hs, he = torch.cuda.Event(True), torch.cuda.Event(True); hs.record(stream)
            tensor = host.to(self.device, dtype=dtype, non_blocking=cuda).mul_(1 / 255.0)
            if cuda:
                he.record(stream); ins, ine = torch.cuda.Event(True), torch.cuda.Event(True); ins.record(stream)
            infer_wall = time.perf_counter(); predictions = predictor.inference(tensor)
            if cuda: ine.record(stream);ine.synchronize()
            infer_wall_ms = (time.perf_counter() - infer_wall) * 1000
            post = time.perf_counter()
            results = predictor.postprocess(predictions, tensor, [p.frame for p in prepared.batch.frames])
            boxes = [np.empty((0, 6), np.float32) if r.boxes is None or len(r.boxes) == 0
                     else r.boxes.data.detach().cpu().numpy() for r in results]
            post_ms = (time.perf_counter() - post) * 1000
        return boxes, {"h2d_ms": hs.elapsed_time(he) if cuda else 0.0,
                       "gpu_inference_ms": ins.elapsed_time(ine) if cuda else infer_wall_ms,
                       "postprocess_ms": post_ms}

    def close(self): self.model = None

class PersonDetector:
    def __init__(self, config, backend=None, max_frame_age_ms=None, max_batch_size=6):
        cfg = config.get("ai", {}).get("detector", {}); raw = cfg.get("imgsz", (448, 800))
        self.input_size = tuple(raw) if isinstance(raw, (list, tuple)) else (raw, raw)
        self.max_frame_age_ms = float(max_frame_age_ms or config.get("ai", {}).get("max_frame_age_ms", 120))
        self.max_det = int(cfg.get("max_det", 50)); self.min_w = float(cfg.get("min_box_width", 3)); self.min_h = float(cfg.get("min_box_height", 6))
        self.low_conf = float(cfg.get("low_conf_size_threshold", .22)); self.low_w = float(cfg.get("low_conf_min_width", 12)); self.low_h = float(cfg.get("low_conf_min_height", 36))
        self.preprocessor = BatchPreprocessor(self.input_size)
        self.backend = backend or UltralyticsBackend(config, max_batch_size)
        self.metrics = DetectorMetrics(); self._lock = threading.Lock()
        self._seen, self._seen_order = set(), deque(maxlen=10000)

    def process_batch(self, batch: BatchOutput):
        started = time.time(); accepted = []
        with self._lock:
            now = time.time()
            for packet in batch.frames:
                key = (packet.camera_id, packet.frame_id)
                if key in self._seen: self.metrics.duplicate_inference_prevented += 1; continue
                if (now - packet.receive_timestamp) * 1000 > self.max_frame_age_ms:
                    self.metrics.stale_drops_before_inference += 1; continue
                if len(self._seen_order) == self._seen_order.maxlen: self._seen.discard(self._seen_order[0])
                self._seen.add(key); self._seen_order.append(key); accepted.append(packet)
            if not accepted: return DetectionBatchResult(batch.batch_id, started, time.time(), ())
            fresh = BatchOutput(batch.batch_id, batch.created_timestamp, tuple(accepted)); wall = time.perf_counter()
            prepared = self.preprocessor.prepare(fresh); raw_results, timing = self.backend.infer(prepared)
            parse = time.perf_counter(); camera_results = []
            for packet, transform, rows in zip(fresh.frames, prepared.transforms, raw_results):
                detections = []
                for row in rows:
                    if len(row) < 6 or int(row[5]) != 0: continue
                    bbox = tuple(float(v) for v in row[:4])
                    if not getattr(self.backend, "coordinates_original", False): bbox = original_bbox(bbox, transform)
                    else:
                        x1, y1, x2, y2 = bbox; bbox = (min(max(x1, 0), packet.width), min(max(y1, 0), packet.height), min(max(x2, 0), packet.width), min(max(y2, 0), packet.height))
                    width, height, confidence = bbox[2] - bbox[0], bbox[3] - bbox[1], float(row[4])
                    if width < self.min_w or height < self.min_h: continue
                    if confidence < self.low_conf and (width < self.low_w or height < self.low_h): continue
                    ratio = width / max(height, 1)
                    if ratio < .08 or ratio > 8: continue
                    detections.append(Detection(bbox, confidence))
                    if len(detections) >= self.max_det: break
                camera_results.append(CameraDetectionResult(packet.camera_id, packet.frame_id, packet.capture_timestamp, packet.receive_timestamp, tuple(detections)))
            parse_ms = (time.perf_counter() - parse) * 1000; completed = time.time(); wall_ms = (time.perf_counter() - wall) * 1000
            self.metrics.record(DetectorBatchMetrics(batch.batch_id, len(accepted), prepared.preprocess_ms, prepared.cpu_pack_ms,
                float(timing.get("h2d_ms", 0)), float(timing.get("gpu_inference_ms", 0)), float(timing.get("postprocess_ms", 0)), parse_ms, wall_ms,
                (completed - min(p.capture_timestamp for p in accepted)) * 1000))
            return DetectionBatchResult(batch.batch_id, started, completed, tuple(camera_results))

    def close(self):
        close = getattr(self.backend, "close", None)
        if close: close()
