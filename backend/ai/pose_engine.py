import threading
import numpy as np

from backend.core.logger import get_logger

log = get_logger("ai.pose")


class PoseEngine:
    """
    YOLO-Pose engine.
    Ankle keypoints: COCO indices 15 and 16.
    """

    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def __init__(self, config):
        self.enabled = bool(config.get("ai", {}).get("pose", {}).get("enabled", True))
        self.model = None
        self.available = False
        self.lock = threading.Lock()

        self.conf = float(config.get("ai", {}).get("pose", {}).get("conf", 0.45))
        self.imgsz = int(config.get("ai", {}).get("pose", {}).get("imgsz", 640))

        model_name = config.get("ai", {}).get("pose", {}).get("model", "models/yolov8n-pose.pt")

        if not self.enabled:
            log.info("Pose engine disabled")
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(model_name)
            self.available = True
            log.info("Pose engine loaded: %s", model_name)

        except Exception as e:
            log.error("Pose engine unavailable: %s", e)

    def detect(self, bgr):
        """
        Returns:
            boxes: list of [x1, y1, x2, y2, conf]  ← conf QO'SHILDI
            keypoints: list of np.array shape (17,2) or None
        """
        if not self.enabled or not self.available or self.model is None or bgr is None:
            return [], []

        with self.lock:
            try:
                results = self.model(
                    bgr,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    verbose=False,
                )
            except Exception as e:
                log.error("Pose inference error: %s", e)
                return [], []

        boxes = []
        keypoints = []

        for r in results:
            try:
                if r.boxes is None or len(r.boxes) == 0:
                    continue

                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()

                for i in range(len(xyxy)):
                    b = xyxy[i]
                    c = float(confs[i])

                    # ✅ [x1, y1, x2, y2, conf] — tracker kutadigan format
                    boxes.append([
                        float(b[0]),
                        float(b[1]),
                        float(b[2]),
                        float(b[3]),
                        c,
                    ])

                if hasattr(r, "keypoints") and r.keypoints is not None and r.keypoints.xy is not None:
                    kpts = r.keypoints.xy.cpu().numpy()
                    for i in range(len(kpts)):
                        keypoints.append(np.array(kpts[i], dtype=np.float32))
                else:
                    for _ in range(len(xyxy)):
                        keypoints.append(None)

            except Exception as e:
                log.error("Pose parse error: %s", e)
                continue

        return boxes, keypoints

    def ankle_point(self, kpt):
        """
        Return average ankle point from left/right ankle.
        """

        if kpt is None:
            return None

        try:
            left = kpt[self.LEFT_ANKLE]
            right = kpt[self.RIGHT_ANKLE]

            points = []

            if left is not None and left[0] > 0 and left[1] > 0:
                points.append(left)

            if right is not None and right[0] > 0 and right[1] > 0:
                points.append(right)

            if not points:
                return None

            arr = np.array(points, dtype=np.float32)
            return float(arr[:, 0].mean()), float(arr[:, 1].mean())

        except Exception:
            return None