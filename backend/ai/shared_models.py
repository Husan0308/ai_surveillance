"""
Shared Models - barcha AIWorker lar uchun markaziy model cache.
"""
import threading
from backend.core.logger import get_logger
from backend.core.gpu_utils import resolve_torch_device

log = get_logger("ai.shared")

_detector = None
_pose = None
_lock = threading.Lock()


def get_detector(model_path, config):
    """Shared YOLO detector - bitta model barcha worker lar uchun."""
    global _detector
    with _lock:
        if _detector is None:
            from ultralytics import YOLO
            _detector = YOLO(model_path)
            _detector.to('cuda')
            _detector.eval()  # inference mode
            log.info(f"Shared detector loaded on GPU: {model_path}")
        return _detector


def get_pose(model_path, config):
    """Shared YOLO-Pose - bitta model barcha worker lar uchun."""
    requested = config.get("ai", {}).get("pose", {}).get("device", "auto")
    device = resolve_torch_device(requested)
    global _pose
    with _lock:
        if _pose is None:
            from ultralytics import YOLO
            _pose = YOLO(model_path)
            _pose.to(device)
            _pose.eval()
            log.info("Shared pose loaded on %s: %s", device, model_path)
        return _pose
