import threading
import numpy as np
import cv2

from backend.core.logger import get_logger

log = get_logger("ai.reid")


class ReIDEngine:
    """
    Optional ReID engine.

    Uses ResNet18 feature extractor if torchvision is available.
    Can be replaced with OSNet later without changing AIWorker.
    """

    def __init__(self, config):
        self.enabled = bool(config.get("ai.reid.enabled", False))
        self.threshold = float(config.get("ai.reid.threshold", 0.65))

        self.model = None
        self.available = False
        self.lock = threading.Lock()

        if not self.enabled:
            log.info("ReID disabled")
            return

        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            self.torch = torch
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            model = models.resnet18(weights="DEFAULT")
            model.fc = torch.nn.Identity()
            model.eval()

            self.model = model
            self.available = True

            log.info("ReID engine loaded: ResNet18")

        except Exception as e:
            log.error("ReID unavailable: %s", e)

    def embed(self, crop_bgr):
        if not self.enabled or not self.available or crop_bgr is None:
            return None

        if crop_bgr.size == 0:
            return None

        try:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transforms(crop_rgb).unsqueeze(0)

            with self.lock:
                with self.torch.no_grad():
                    feat = self.model(tensor)

            feat = feat.squeeze().cpu().numpy().astype(np.float32)
            n = np.linalg.norm(feat)

            if n == 0:
                return None

            return feat / n

        except Exception as e:
            log.error("ReID embed error: %s", e)
            return None

    @staticmethod
    def similarity(a, b):
        if a is None or b is None:
            return 0.0

        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        if na == 0 or nb == 0:
            return 0.0

        return float(np.dot(a, b) / (na * nb))