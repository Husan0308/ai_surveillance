from __future__ import annotations

import mmap
import os
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path


MAGIC = b"AISMMAP1"
HEADER_SIZE = 64
SEQ_OFFSET = 8
META_OFFSET = 16
_U64 = struct.Struct("<Q")
# active_slot, width, height, channels, payload_bytes, frame_id,
# captured_monotonic_ns, published_monotonic_ns
_META = struct.Struct("<IIIIQQQQ")


def _safe_camera_token(camera_id: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(camera_id)).strip("_")
    return token or "camera"


def frame_directory() -> Path:
    configured = os.environ.get("AI_SURVEILLANCE_FRAME_DIR", "").strip()
    if configured:
        root = Path(configured).expanduser()
    else:
        shm = Path("/dev/shm")
        root = shm if shm.is_dir() and os.access(shm, os.W_OK) else Path("/tmp")
    root.mkdir(parents=True, exist_ok=True)
    return root


def frame_path(camera_id: str) -> Path:
    return frame_directory() / f"ai_surveillance_{_safe_camera_token(camera_id)}.frame"


@dataclass(frozen=True, slots=True)
class MmapFrame:
    sequence: int
    width: int
    height: int
    channels: int
    frame_id: int
    captured_monotonic_ns: int
    published_monotonic_ns: int
    payload: bytes

    @property
    def age_ms(self) -> float:
        if self.published_monotonic_ns <= 0:
            return 0.0
        return max(0.0, (time.monotonic_ns() - self.published_monotonic_ns) / 1_000_000.0)


class MmapFrameWriter:
    """Latest-only double-buffered BGR frame writer for the local Qt UI.

    A tiny seqlock protects readers from seeing a half-written frame:
    sequence is odd while the inactive slot is being written and becomes even
    only after frame bytes + metadata are complete. No queue exists, so slow UI
    readers always jump directly to the newest complete frame.
    """

    def __init__(self, camera_id: str, max_width: int, max_height: int, channels: int = 3):
        self.camera_id = str(camera_id)
        self.max_width = max(1, int(max_width))
        self.max_height = max(1, int(max_height))
        self.channels = max(1, int(channels))
        self.slot_bytes = self.max_width * self.max_height * self.channels
        self.total_bytes = HEADER_SIZE + 2 * self.slot_bytes
        self.path = frame_path(self.camera_id)
        self._fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
        os.ftruncate(self._fd, self.total_bytes)
        self._map = mmap.mmap(self._fd, self.total_bytes, access=mmap.ACCESS_WRITE)
        self._sequence = 0
        self._active_slot = 0
        self._closed = False
        self._map[0:8] = MAGIC
        self._map[SEQ_OFFSET:SEQ_OFFSET + 8] = _U64.pack(0)
        self._map[META_OFFSET:HEADER_SIZE] = _META.pack(0, 0, 0, self.channels, 0, 0, 0, 0)

    def write(self, image, frame_id: int, captured_monotonic: float) -> dict:
        if self._closed:
            raise RuntimeError("mmap frame writer is closed")
        if image is None or getattr(image, "ndim", 0) != 3:
            raise ValueError("expected HxWxC image")
        height, width, channels = [int(value) for value in image.shape[:3]]
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")
        if width > self.max_width or height > self.max_height:
            raise ValueError(
                f"frame {width}x{height} exceeds mmap slot {self.max_width}x{self.max_height}"
            )

        payload = image.tobytes(order="C")
        payload_bytes = len(payload)
        if payload_bytes > self.slot_bytes:
            raise ValueError("frame payload exceeds mmap slot")

        inactive_slot = 1 - self._active_slot
        odd_sequence = self._sequence + 1
        even_sequence = self._sequence + 2

        # Mark write in progress before touching the inactive frame slot.
        self._map[SEQ_OFFSET:SEQ_OFFSET + 8] = _U64.pack(odd_sequence)
        slot_offset = HEADER_SIZE + inactive_slot * self.slot_bytes
        self._map[slot_offset:slot_offset + payload_bytes] = payload

        published_ns = time.monotonic_ns()
        captured_ns = max(0, int(float(captured_monotonic) * 1_000_000_000.0))
        self._map[META_OFFSET:HEADER_SIZE] = _META.pack(
            inactive_slot,
            width,
            height,
            channels,
            payload_bytes,
            max(0, int(frame_id)),
            captured_ns,
            published_ns,
        )
        # Commit is last. Readers only accept identical even seq values before
        # and after copying the selected slot.
        self._map[SEQ_OFFSET:SEQ_OFFSET + 8] = _U64.pack(even_sequence)
        self._sequence = even_sequence
        self._active_slot = inactive_slot
        return {
            "sequence": even_sequence,
            "width": width,
            "height": height,
            "payload_bytes": payload_bytes,
            "frame_id": int(frame_id),
            "published_monotonic_ns": published_ns,
        }

    def close(self, unlink: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._map.close()
        finally:
            try:
                os.close(self._fd)
            finally:
                if unlink:
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass


class MmapFrameReader:
    """Attach to one writer and copy only complete latest frames."""

    def __init__(self, camera_id: str):
        self.camera_id = str(camera_id)
        self.path = frame_path(self.camera_id)
        self._fd: int | None = None
        self._map: mmap.mmap | None = None
        self._inode: int | None = None
        self._size = 0

    def attach(self) -> bool:
        self.close()
        try:
            fd = os.open(str(self.path), os.O_RDONLY)
            stat = os.fstat(fd)
            if stat.st_size < HEADER_SIZE + 2:
                os.close(fd)
                return False
            mapped = mmap.mmap(fd, stat.st_size, access=mmap.ACCESS_READ)
            if mapped[0:8] != MAGIC:
                mapped.close()
                os.close(fd)
                return False
        except (FileNotFoundError, OSError, ValueError):
            return False
        self._fd = fd
        self._map = mapped
        self._inode = int(stat.st_ino)
        self._size = int(stat.st_size)
        return True

    def mapping_is_current(self) -> bool:
        if self._map is None or self._fd is None or self._inode is None:
            return False
        try:
            return int(os.stat(self.path).st_ino) == self._inode
        except OSError:
            return False

    def snapshot(self, last_sequence: int | None = None) -> MmapFrame | None:
        mapped = self._map
        if mapped is None and not self.attach():
            return None
        mapped = self._map
        if mapped is None:
            return None

        try:
            sequence_before = _U64.unpack_from(mapped, SEQ_OFFSET)[0]
            if sequence_before == 0 or sequence_before & 1:
                return None
            if last_sequence is not None and sequence_before == int(last_sequence):
                return None

            active_slot, width, height, channels, payload_bytes, frame_id, captured_ns, published_ns = (
                _META.unpack_from(mapped, META_OFFSET)
            )
            if active_slot not in (0, 1) or width <= 0 or height <= 0 or channels <= 0:
                return None
            expected_bytes = int(width) * int(height) * int(channels)
            if payload_bytes != expected_bytes:
                return None
            slot_bytes = (self._size - HEADER_SIZE) // 2
            if payload_bytes <= 0 or payload_bytes > slot_bytes:
                return None

            offset = HEADER_SIZE + int(active_slot) * slot_bytes
            payload = bytes(mapped[offset:offset + int(payload_bytes)])
            sequence_after = _U64.unpack_from(mapped, SEQ_OFFSET)[0]
            if sequence_before != sequence_after or sequence_after & 1:
                return None
        except (ValueError, BufferError, OSError, struct.error):
            return None

        return MmapFrame(
            sequence=int(sequence_after),
            width=int(width),
            height=int(height),
            channels=int(channels),
            frame_id=int(frame_id),
            captured_monotonic_ns=int(captured_ns),
            published_monotonic_ns=int(published_ns),
            payload=payload,
        )

    def close(self) -> None:
        mapped, fd = self._map, self._fd
        self._map = None
        self._fd = None
        self._inode = None
        self._size = 0
        if mapped is not None:
            try:
                mapped.close()
            except (BufferError, OSError):
                pass
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
