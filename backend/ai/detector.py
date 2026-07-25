import threading

from backend.core.logger import get_logger

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

        self.conf = float(config.get("ai", {}).get("detector", {}).get("conf", 0.45))
        self.imgsz = int(config.get("ai", {}).get("detector", {}).get("imgsz", 640))
        self.device = config.get("ai", {}).get("detector", {}).get("device", "auto")

        model_name = config.get("ai", {}).get("detector", {}).get("model", "models/yolov8n.pt")

        try:
            from ultralytics import YOLO

            self.model = YOLO(model_name)

            if self.device and self.device != "auto":
                try:
                    self.model.to(self.device)
                except Exception:
                    pass

            self.available = True
            log.info("Detector loaded: %s", model_name)

        except Exception as e:
            log.error("Detector unavailable: %s", e)

    def detect(self, bgr):
        if not self.available or self.model is None or bgr is None:
            return []

        with self.lock:
            try:
                results = self.model(
                    bgr,
                    classes=[0],
                    conf=self.conf,
                    imgsz=self.imgsz,
                    verbose=False,
                )
            except Exception as e:
                log.error("Detector inference error: %s", e)
                return []

        boxes = []

        for r in results:
            try:
                if r.boxes is None or len(r.boxes) == 0:
                    continue

                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()

                for b, c in zip(xyxy, confs):
                    boxes.append([
                        float(b[0]),
                        float(b[1]),
                        float(b[2]),
                        float(b[3]),
                        float(c),
                    ])
            except Exception:
                continue

        return boxes