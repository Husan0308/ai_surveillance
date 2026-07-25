import os
import time
import cv2

from backend.cameras.utils import build_source_url, is_int_source


def test_connection(source, username=None, password=None, timeout: int = 5) -> dict:
    """
    Kamera manbasini tekshiradi.

    Qaytaradi:
        {
            "ok": bool,
            "message": str,
            "latency_ms": float,
            "resolution": str,
            "fps": float
        }
    """

    result = {
        "ok": False,
        "message": "",
        "latency_ms": None,
        "resolution": None,
        "fps": None,
    }

    src = build_source_url(source, username, password)

    try:
        if is_int_source(src):
            api = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(int(src), api)
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)

            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout * 1000))
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout * 1000))
            except Exception:
                pass

        if cap is None or not cap.isOpened():
            result["message"] = "Cannot open camera source"
            if cap is not None:
                cap.release()
            return result

        start = time.time()

        while time.time() - start < timeout:
            t0 = time.time()
            ret, frame = cap.read()

            if ret and frame is not None:
                result["ok"] = True
                result["message"] = "Connected"
                result["latency_ms"] = round((time.time() - t0) * 1000.0, 1)
                result["resolution"] = f"{frame.shape[1]}x{frame.shape[0]}"

                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps and fps > 0:
                    result["fps"] = round(float(fps), 1)

                break

            time.sleep(0.05)

        if not result["ok"]:
            result["message"] = "No frame received"

        cap.release()
        return result

    except Exception as e:
        result["message"] = f"Connection error: {e}"
        return result


if __name__ == "__main__":
    from backend.core.config import ConfigService

    cfg = ConfigService()
    cams = cfg.load_cameras()

    if not cams:
        print("No cameras found in config/cameras.yaml")
    else:
        for cam in cams:
            print("\nTesting:", cam.get("id"), cam.get("source"))
            r = test_connection(
                cam.get("source"),
                cam.get("username"),
                cam.get("password"),
                timeout=6,
            )
            print(r)