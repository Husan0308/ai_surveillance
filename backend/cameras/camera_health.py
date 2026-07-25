import time


class CameraHealth:
    def __init__(self):
        self.reset()

    def reset(self):
        self.online = False
        self.latency = 0.0
        self.fps = 0.0
        self.total_reads = 0
        self.failed_reads = 0
        self.last_frame_ts = 0.0

    def record_read(self, ok: bool, latency_ms: float):
        self.total_reads += 1

        if ok:
            self.last_frame_ts = time.time()

            if self.latency == 0:
                self.latency = latency_ms
            else:
                self.latency = (self.latency * 0.8) + (latency_ms * 0.2)
        else:
            self.failed_reads += 1

    def set_fps(self, fps: float):
        self.fps = float(fps)

    @property
    def packet_loss(self) -> float:
        if self.total_reads <= 0:
            return 0.0
        return (self.failed_reads / self.total_reads) * 100.0

    @property
    def last_frame_age(self) -> float:
        if self.last_frame_ts <= 0:
            return 0.0
        return time.time() - self.last_frame_ts

    @property
    def conn_quality(self) -> int:
        """
        0..4 signal bars
        """

        if not self.online:
            return 0

        score = 4

        if self.latency > 50:
            score -= 1
        if self.latency > 30:
            score -= 1
        if self.packet_loss > 2:
            score -= 1
        if self.packet_loss > 1:
            score -= 1

        return max(1, score)

    def metrics(self) -> dict:
        return {
            "online": self.online,
            "latency_ms": round(self.latency, 1),
            "fps": round(self.fps, 1),
            "packet_loss": round(self.packet_loss, 2),
            "conn_quality": self.conn_quality,
            "last_frame_age": round(self.last_frame_age, 2),
            "total_reads": self.total_reads,
            "failed_reads": self.failed_reads,
        }