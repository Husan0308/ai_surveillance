from dataclasses import dataclass
from services.ml_service.cameras.frame import FramePacket

@dataclass(frozen=True, slots=True)
class BatchOutput:
    batch_id: int
    created_timestamp: float
    frames: tuple[FramePacket, ...]

    @property
    def camera_ids(self): return tuple(packet.camera_id for packet in self.frames)
    def detector_frames(self): return [packet.frame for packet in self.frames]
