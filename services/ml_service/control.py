"""Small internal ML control/status/video API."""
import queue, threading, time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

class MLRuntimeState:
    def __init__(self):
        self.commands = queue.Queue(64)
        self.metrics = {}
        self.status = {"status": "starting"}
        self.frames = {}
        self.lock = threading.Lock()

    def command(self, item):
        try:
            self.commands.put_nowait(item)
        except queue.Full:
            raise HTTPException(429, "ML command queue full")

    def poll(self):
        result = []
        while True:
            try:
                result.append(self.commands.get_nowait())
            except queue.Empty:
                return result

    def frame(self, packet):
        with self.lock:
            self.frames[packet.camera_id] = (packet.frame_id, packet.receive_timestamp, packet.frame)


runtime = MLRuntimeState()
app = FastAPI(title="Internal ML Control API")


@app.get("/health")
def health():
    return {"service": "ml-service", "status": runtime.status.get("status", "unknown")}


@app.get("/ready")
def ready():
    # Check if detector is initialized and models are ready
    status = runtime.status.get("status", "unknown")
    models_ready = status == "running" and runtime.metrics.get("detector_ms", 0) > 0
    return {
        "service": "ml-service",
        "status": status,
        "ready": models_ready,
        "cameras_active": runtime.status.get("cameras", 0)
    }


@app.get("/status")
def status():
    return runtime.status


@app.get("/metrics")
def metrics():
    return runtime.metrics


@app.post("/commands")
def command(message: dict):
    if not isinstance(message.get("type"), str):
        raise HTTPException(422, "Command type is required")
    runtime.command(message)
    return {"accepted": True}


@app.get("/video/{camera_id}")
def video(camera_id: str):
    def stream():
        import cv2

        last = -1
        while True:
            with runtime.lock:
                item = runtime.frames.get(camera_id)
            if item is None or item[0] == last:
                time.sleep(.015)
                continue
            last = item[0]
            ok, encoded = cv2.imencode(".jpg", item[2], [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nX-Frame-Id: " + str(item[0]).encode() + b"\r\nX-Timestamp: " + str(item[1]).encode() + b"\r\n\r\n" + encoded.tobytes() + b"\r\n"

    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")
