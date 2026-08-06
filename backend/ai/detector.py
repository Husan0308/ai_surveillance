import threading
import torch

from backend.core.logger import get_logger
from backend.core.gpu_utils import resolve_torch_device

log = get_logger("ai.detector")


class Detector:
    """
    YOLO person detector.
    Shared model. Thread-safe.
    """

    def __init__(self, config):
        self.model = None
        self.available = False
        self.lock = threading.Lock()

        # ASOSIY FIX: YOLO conf qiymatini ByteTrack uchun 0.10 ga tushiramiz.
        # Bu orqali qisman to'silib qolgan yoki uzoqdagi odamlarni tracker ushlab qoladi 
        # va ramkalar "sakrab" ketishining oldi olinadi.
        self.conf = float(config.get("ai", {}).get("detector", {}).get("conf", 0.30))
        
        self.imgsz = (384, 640)
        requested = config.get("ai", {}).get("detector", {}).get("device", "auto")
        self.device = resolve_torch_device(requested)
        model_name = config.get("ai", {}).get("detector", {}).get("model", "models/yolov8n.pt")

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            # Move model to the resolved device only if it's a PyTorch model
            if model_name.endswith((".pt", ".pth")):
                self.model.to(self.device)
            self.available = True
            print(f"[Detector] Loaded {model_name} on {self.device} with conf={self.conf}", flush=True)
        except Exception as e:
            print(f"[Detector] ❌ Failed to load model {model_name}: {e}", flush=True)
            self.available = False

    def detect(self, bgr):
        res = self.detect_batch([bgr])
        return res[0] if res else []

    def detect_batch(self, bgr_list):
        if not self.available or self.model is None or not bgr_list:
            return [[] for _ in bgr_list]

        valid_inputs = []
        valid_indices = []
        for i, img in enumerate(bgr_list):
            if img is not None and img.size > 0:
                valid_inputs.append(img)
                valid_indices.append(i)

        batch_results = [[] for _ in bgr_list]
        if not valid_inputs:
            return batch_results

        with self.lock:
            try:
                with torch.inference_mode():
                    results = self.model(
                        valid_inputs,
                        classes=[0],
                        conf=self.conf,
                        iou=0.45,  # GPU-accelerated NMS
                        imgsz=self.imgsz,
                        device=self.device,
                        batch=len(valid_inputs),
                        verbose=False,
                    )
            except Exception as e:
                log.error("Detector batch inference error: %s", e)
                return batch_results

        for idx, r in zip(valid_indices, results):
            boxes = []
            try:
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    for b, c in zip(xyxy, confs):
                        w = b[2] - b[0]
                        h = b[3] - b[1]
                        
                        # Minimallashtirilgan filter (uzoqdagi odamlarni o'chirib yubormaslik uchun)
                        if w < 4 or h < 8:
                            continue
                            
                        ratio = w / max(h, 1)
                        if ratio < 0.08 or ratio > 8.0:
                            continue

                        boxes.append([
                            float(b[0]),
                            float(b[1]),
                            float(b[2]),
                            float(b[3]),
                            float(c),
                        ])
            except Exception as e:
                log.error("Error processing bbox results: %s", e)
            batch_results[idx] = boxes

        return batch_results
