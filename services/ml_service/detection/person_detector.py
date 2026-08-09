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

def filter_end2end_predictions(array,conf,classes,max_det):
    """Equivalent to Ultralytics' end-to-end NMS branch, after one batch D2H copy."""
    allowed=None if classes is None else np.asarray(classes,dtype=np.int64)
    output=[]
    for rows in array:
        selected=rows[rows[:,4]>float(conf)][:max_det]
        if allowed is not None:selected=selected[np.isin(selected[:,5].astype(np.int64,copy=False),allowed)]
        output.append(np.ascontiguousarray(selected[:,:6],dtype=np.float32))
    return output

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
        from ultralytics.utils import nms
        cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")
        dtype = torch.float16 if getattr(predictor.model, "fp16", False) else torch.float32
        with torch.inference_mode():
            phase=time.perf_counter();host = torch.from_numpy(prepared.images_nchw);numpy_to_torch_ms=(time.perf_counter()-phase)*1000
            if cuda:
                phase=time.perf_counter();host = host.pin_memory();pin_memory_ms=(time.perf_counter()-phase)*1000; stream = torch.cuda.current_stream(self.device)
                hs, he = torch.cuda.Event(True), torch.cuda.Event(True); hs.record(stream)
            else:pin_memory_ms=0.0
            phase=time.perf_counter()
            tensor = host.to(self.device, dtype=dtype, non_blocking=cuda).mul_(1 / 255.0)
            h2d_wall_ms=(time.perf_counter()-phase)*1000
            if cuda:
                he.record(stream); ins, ine = torch.cuda.Event(True), torch.cuda.Event(True); ins.record(stream)
            infer_wall = time.perf_counter(); predictions = predictor.inference(tensor)
            if cuda: ine.record(stream);ine.synchronize()
            infer_wall_ms = (time.perf_counter() - infer_wall) * 1000
            end2end=bool(getattr(predictor.model,"end2end",False))
            if end2end:
                raw=predictions[0] if isinstance(predictions,(list,tuple)) else predictions
                phase=time.perf_counter();host_predictions=raw.detach().cpu().numpy();tensor_to_cpu_ms=(time.perf_counter()-phase)*1000
                phase=time.perf_counter();boxes=filter_end2end_predictions(host_predictions,predictor.args.conf,predictor.args.classes,predictor.args.max_det);nms_ms=(time.perf_counter()-phase)*1000
                results_ms=0.0;camera_copy_ms=[];self.coordinates_original=False
            else:
                phase=time.perf_counter();preds=nms.non_max_suppression(predictions,predictor.args.conf,predictor.args.iou,
                    predictor.args.classes,predictor.args.agnostic_nms,max_det=predictor.args.max_det,
                    nc=0 if predictor.args.task=="detect" else len(predictor.model.names),end2end=False,rotated=False)
                nms_ms=(time.perf_counter()-phase)*1000
                phase=time.perf_counter();results=predictor.construct_results(preds,tensor,[p.frame for p in prepared.batch.frames]);results_ms=(time.perf_counter()-phase)*1000
                boxes=[];camera_copy_ms=[]
                for result in results:
                    copy_started=time.perf_counter();boxes.append(np.empty((0,6),np.float32) if result.boxes is None or len(result.boxes)==0 else result.boxes.data.detach().cpu().numpy());camera_copy_ms.append((time.perf_counter()-copy_started)*1000)
                tensor_to_cpu_ms=sum(camera_copy_ms);self.coordinates_original=True
            post_ms=nms_ms+results_ms+tensor_to_cpu_ms
        return boxes, {"h2d_ms": hs.elapsed_time(he) if cuda else 0.0,
                       "gpu_inference_ms": ins.elapsed_time(ine) if cuda else infer_wall_ms,
                       "postprocess_ms":post_ms,"numpy_to_torch_ms":numpy_to_torch_ms,"pin_memory_ms":pin_memory_ms,
                       "h2d_wall_ms":h2d_wall_ms,"model_forward_wall_ms":infer_wall_ms,"nms_ms":nms_ms,
                       "results_construction_ms":results_ms,"tensor_to_cpu_ms":tensor_to_cpu_ms,
                       "camera_tensor_to_cpu_ms":camera_copy_ms}

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
        started = time.time(); accepted = [];validation_started=time.perf_counter()
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
            validation_ms=(time.perf_counter()-validation_started)*1000
            fresh = BatchOutput(batch.batch_id, batch.created_timestamp, tuple(accepted)); wall = time.perf_counter()
            prepared = self.preprocessor.prepare(fresh); raw_results, timing = self.backend.infer(prepared)
            parse = time.perf_counter();camera_results=[];filter_ms=rescale_ms=result_build_ms=0.0
            for packet, transform, rows in zip(fresh.frames, prepared.transforms, raw_results):
                detections=[];phase=time.perf_counter();camera_rescale_ms=0.0
                for row in rows:
                    if len(row) < 6 or int(row[5]) != 0: continue
                    bbox = tuple(float(v) for v in row[:4])
                    if not getattr(self.backend, "coordinates_original", False):
                        rescale_started=time.perf_counter();bbox=original_bbox(bbox,transform);value=(time.perf_counter()-rescale_started)*1000;rescale_ms+=value;camera_rescale_ms+=value
                    else:
                        x1, y1, x2, y2 = bbox; bbox = (min(max(x1, 0), packet.width), min(max(y1, 0), packet.height), min(max(x2, 0), packet.width), min(max(y2, 0), packet.height))
                    width, height, confidence = bbox[2] - bbox[0], bbox[3] - bbox[1], float(row[4])
                    if width < self.min_w or height < self.min_h: continue
                    if confidence < self.low_conf and (width < self.low_w or height < self.low_h): continue
                    ratio = width / max(height, 1)
                    if ratio < .08 or ratio > 8: continue
                    detections.append(Detection(bbox, confidence))
                    if len(detections) >= self.max_det: break
                filter_ms+=max(0.0,(time.perf_counter()-phase)*1000-camera_rescale_ms)
                phase=time.perf_counter();camera_results.append(CameraDetectionResult(packet.camera_id,packet.frame_id,packet.capture_timestamp,packet.receive_timestamp,tuple(detections)));result_build_ms+=(time.perf_counter()-phase)*1000
            parse_ms = (time.perf_counter() - parse) * 1000; completed = time.time(); wall_ms = (time.perf_counter() - wall) * 1000
            phases={"frame_validation":validation_ms,"preprocess_total":prepared.preprocess_ms,"postprocess_total":float(timing.get("postprocess_ms",0))+parse_ms,"image_resize":prepared.resize_ms,"letterbox":prepared.letterbox_ms,
                "bgr_to_rgb":prepared.bgr_to_rgb_ms,"numpy_stacking":prepared.numpy_stack_ms,
                "numpy_to_torch":float(timing.get("numpy_to_torch_ms",0)),"pinned_memory":float(timing.get("pin_memory_ms",0)),
                "cpu_to_gpu":float(timing.get("h2d_ms",0)),"model_forward":float(timing.get("gpu_inference_ms",0)),
                "nms":float(timing.get("nms_ms",0)),"results_construction":float(timing.get("results_construction_ms",0)),
                "tensor_to_cpu":float(timing.get("tensor_to_cpu_ms",0)),"bbox_filtering":max(0.0,filter_ms),
                "coordinate_rescaling":rescale_ms,"camera_result_construction":result_build_ms,"pure_detector_wall":wall_ms}
            self.metrics.record(DetectorBatchMetrics(batch.batch_id, len(accepted), prepared.preprocess_ms, prepared.cpu_pack_ms,
                float(timing.get("h2d_ms", 0)), float(timing.get("gpu_inference_ms", 0)), float(timing.get("postprocess_ms", 0)), parse_ms, wall_ms,
                (completed-min(p.capture_timestamp for p in accepted))*1000,phases))
            return DetectionBatchResult(batch.batch_id, started, completed, tuple(camera_results))

    def close(self):
        close = getattr(self.backend, "close", None)
        if close: close()
