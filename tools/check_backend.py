import os
import sys
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.core.config import ConfigService
from backend.cameras.utils import build_source_url
from backend.ai.detector import Detector
from backend.ai.pose_engine import PoseEngine


def main():
    cfg = ConfigService()

    cams = cfg.load_cameras()

    if cams:
        cam = cams[0]
        source = cam.get("source", 0)
        username = cam.get("username")
        password = cam.get("password")
    else:
        source = 0
        username = None
        password = None

    url = build_source_url(source, username, password)

    print("Camera source:", source)
    print("RTSP URL:", url)

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    if isinstance(url, int):
        cap = cv2.VideoCapture(url)
    else:
        cap = cv2.VideoCapture(str(url), cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("❌ Kamera ochilmadi")
        return

    ret, frame = cap.read()

    cap.release()

    if not ret or frame is None:
        print("❌ Frame o‘qilmadi")
        return

    print("✅ Frame:", frame.shape)

    print("\n--- YOLO Detector ---")

    detector = Detector(cfg)

    print("Detector available:", detector.available)

    if detector.available:
        boxes = detector.detect(frame)
        print("YOLO boxes:", len(boxes))

        for i, b in enumerate(boxes[:5]):
            print(f"  box {i}: {[round(x, 1) for x in b]}")

    print("\n--- YOLO Pose ---")

    pose = PoseEngine(cfg)

    print("Pose enabled:", pose.enabled)
    print("Pose available:", pose.available)

    if pose.available:
        pboxes, kpts = pose.detect(frame)

        print("Pose boxes:", len(pboxes))

        ankle_count = 0

        for k in kpts:
            ankle = pose.ankle_point(k)

            if ankle is not None:
                ankle_count += 1

        print("Ankle points:", ankle_count)

    print("\n✅ Check finished")


if __name__ == "__main__":
    main()