import threading
import numpy as np
import torch

from backend.core.logger import get_logger

log = get_logger("ai.pose")
from backend.core.gpu_utils import resolve_torch_device


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
        requested = config.get("ai", {}).get("pose", {}).get("device", "auto")
        self.device = resolve_torch_device(requested)

        self.conf = float(config.get("ai", {}).get("pose", {}).get("conf", 0.45))
        self.imgsz = int(config.get("ai", {}).get("pose", {}).get("imgsz", 640))

        model_name = config.get("ai", {}).get("pose", {}).get("model", "models/yolov8n-pose.pt")

        if not self.enabled:
            log.info("Pose engine disabled")
            return

        try:
            from backend.ai.shared_models import get_pose
            self.model = get_pose(model_name, config)
            self.available = True
            log.info("Pose engine loaded: %s", model_name)

        except Exception as e:
            log.error("Pose engine unavailable: %s", e)

    def detect(self, bgr):
        res = self.detect_batch([bgr])
        return res[0] if res else ([], [])

    def detect_batch(self, bgr_list):
        if not self.enabled or not self.available or self.model is None or not bgr_list:
            return [([], []) for _ in bgr_list]

        valid_inputs = []
        valid_indices = []
        for i, img in enumerate(bgr_list):
            if img is not None and img.size > 0:
                valid_inputs.append(img)
                valid_indices.append(i)

        batch_results = [([], []) for _ in bgr_list]
        if not valid_inputs:
            return batch_results

        with self.lock:
            try:
                use_cuda = getattr(self.device, "type", str(self.device)) == "cuda"
                with torch.inference_mode():
                    with torch.amp.autocast('cuda', enabled=use_cuda):
                        results = self.model(
                            valid_inputs,
                            classes=[0],
                            conf=self.conf,
                            imgsz=self.imgsz,
                            device=self.device,
                            batch=len(valid_inputs),
                            verbose=False,
                        )
            except Exception as e:
                log.error("Pose batch inference error: %s", e)
                return batch_results

        for idx, r in zip(valid_indices, results):
            boxes = []
            keypoints = []
            try:
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    has_kpts = hasattr(r, "keypoints") and r.keypoints is not None and r.keypoints.xy is not None
                    raw_kpts = r.keypoints.xy.cpu().numpy() if has_kpts else None

                    for i in range(len(xyxy)):
                        b = xyxy[i]
                        c = float(confs[i])
                        w = b[2] - b[0]
                        h = b[3] - b[1]

                        # Sanity check: humans are vertical/seated silhouettes (height >= 30px, width >= 15px)
                        # Filter out thin horizontal lines or tiny noise boxes (width > height * 2.8)
                        if w < 15 or h < 30 or w > h * 2.8:
                            continue

                        boxes.append([
                            float(b[0]),
                            float(b[1]),
                            float(b[2]),
                            float(b[3]),
                            c,
                        ])

                        if raw_kpts is not None and i < len(raw_kpts):
                            keypoints.append(np.array(raw_kpts[i], dtype=np.float32))
                        else:
                            keypoints.append(None)
            except Exception as e:
                log.error("Pose parse error: %s", e)
            batch_results[idx] = (boxes, keypoints)

        return batch_results

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