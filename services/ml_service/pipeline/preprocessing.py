"""One-pass batch letterboxing and packing, independent of model execution."""
from __future__ import annotations
from dataclasses import dataclass
import time
import cv2
import numpy as np
from services.ml_service.pipeline.batch import BatchOutput

@dataclass(frozen=True, slots=True)
class ImageTransform:
    scale: float
    pad_x: float
    pad_y: float
    original_width: int
    original_height: int

@dataclass(frozen=True, slots=True)
class PreparedBatch:
    batch: BatchOutput
    images_nchw: np.ndarray
    transforms: tuple[ImageTransform, ...]
    preprocess_ms: float
    cpu_pack_ms: float
    resize_ms: float = 0.0
    letterbox_ms: float = 0.0
    bgr_to_rgb_ms: float = 0.0
    numpy_stack_ms: float = 0.0
    full_frame_copies: int = 0

class BatchPreprocessor:
    def __init__(self, input_size=(448, 800), stride=32, pad_value=114):
        height, width = input_size
        self.height, self.width = int(height), int(width)
        self.stride, self.pad_value = int(stride), int(pad_value)
        self._buffers: dict[int, np.ndarray] = {}

    def prepare(self, batch: BatchOutput) -> PreparedBatch:
        started = time.perf_counter(); count = len(batch.frames)
        packed = self._buffers.get(count)
        if packed is None:
            packed = np.empty((count, 3, self.height, self.width), dtype=np.uint8)
            self._buffers[count] = packed
        transforms = [];resize_ms=letterbox_ms=bgr_to_rgb_ms=numpy_stack_ms=0.0
        pack_started = time.perf_counter()
        for index, packet in enumerate(batch.frames):
            frame = packet.frame
            scale = min(self.width / packet.width, self.height / packet.height)
            resized_w, resized_h = round(packet.width * scale), round(packet.height * scale)
            pad_x, pad_y = (self.width - resized_w) / 2, (self.height - resized_h) / 2
            if resized_w == packet.width and resized_h == packet.height:
                resized = frame
            else:
                phase=time.perf_counter()
                resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
                resize_ms+=(time.perf_counter()-phase)*1000
            left, top = int(round(pad_x - .1)), int(round(pad_y - .1))
            right, bottom = self.width - resized_w - left, self.height - resized_h - top
            if left or right or top or bottom:
                phase=time.perf_counter()
                resized = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                             cv2.BORDER_CONSTANT, value=(self.pad_value,) * 3)
                letterbox_ms+=(time.perf_counter()-phase)*1000
            phase=time.perf_counter();rgb_chw=resized[...,::-1].transpose(2,0,1);bgr_to_rgb_ms+=(time.perf_counter()-phase)*1000
            phase=time.perf_counter();np.copyto(packed[index],rgb_chw);numpy_stack_ms+=(time.perf_counter()-phase)*1000
            transforms.append(ImageTransform(scale, left, top, packet.width, packet.height))
        cpu_pack_ms = (time.perf_counter() - pack_started) * 1000
        return PreparedBatch(batch,packed,tuple(transforms),(time.perf_counter()-started)*1000,cpu_pack_ms,
                             resize_ms,letterbox_ms,bgr_to_rgb_ms,numpy_stack_ms,0)

def original_bbox(bbox, transform: ImageTransform):
    x1, y1, x2, y2 = bbox
    scale = max(transform.scale, 1e-12)
    x1, x2 = (x1 - transform.pad_x) / scale, (x2 - transform.pad_x) / scale
    y1, y2 = (y1 - transform.pad_y) / scale, (y2 - transform.pad_y) / scale
    x1 = min(max(float(x1), 0.0), float(transform.original_width))
    x2 = min(max(float(x2), 0.0), float(transform.original_width))
    y1 = min(max(float(y1), 0.0), float(transform.original_height))
    y2 = min(max(float(y2), 0.0), float(transform.original_height))
    return x1, y1, x2, y2
