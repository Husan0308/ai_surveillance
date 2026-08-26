from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path


MAGIC = b"CAMSHM01"
VERSION = 1
HEADER_SIZE = 128
# magic, version, seq, slot, width, height, stride, frame_bytes, captured_ns
_HEADER = struct.Struct("<8sIQQIIIIQ")

REQUEST_MAGIC = b"CAMREQ01"
REQUEST_VERSION = 1
REQUEST_SIZE = 64
# Keep request_seq naturally aligned at byte offset 16.
_REQUEST_HEADER = struct.Struct("<8sIIQ")
_REQUEST_SEQ_OFFSET = 16
_REQUEST_SEQ = struct.Struct("<Q")


def _safe_camera_name(camera_id: str) -> str:
    return camera_id.lower().replace("-", "_")


def _shm_root() -> Path:
    root = Path(os.environ.get("CAMERA_SERVICE_SHM_DIR", "/dev/shm/ai_surveillance"))
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass(frozen=True)
class FrameSnapshot:
    seq: int
    slot: int
    width: int
    height: int
    stride: int
    frame_bytes: int
    captured_ns: int
    data: bytes


class LatestFrameMmapWriter:
    """Double-buffered latest-frame transport backed by /dev/shm.

    The writer always fills the inactive slot first and publishes a new even
    sequence number only after the frame bytes are complete. Readers never hold
    a lock and never block the camera pipeline; they simply retry when the header
    changes while copying a slot.
    """

    def __init__(self, camera_id: str, width: int, height: int, channels: int = 3) -> None:
        safe = _safe_camera_name(camera_id)
        self.path = _shm_root() / f"{safe}.frame"
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.stride = self.width * self.channels
        self.frame_bytes = self.stride * self.height
        self.size = HEADER_SIZE + (2 * self.frame_bytes)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(self.fd, self.size)
        self.mm = mmap.mmap(self.fd, self.size, access=mmap.ACCESS_WRITE)
        self.seq = 0
        self.slot = 0
        self._publish_header(captured_ns=0)

    def _publish_header(self, captured_ns: int) -> None:
        _HEADER.pack_into(
            self.mm,
            0,
            MAGIC,
            VERSION,
            int(self.seq),
            int(self.slot),
            self.width,
            self.height,
            self.stride,
            self.frame_bytes,
            int(captured_ns),
        )
        if _HEADER.size < HEADER_SIZE:
            self.mm[_HEADER.size:HEADER_SIZE] = b"\0" * (HEADER_SIZE - _HEADER.size)

    def publish(self, frame_bytes: memoryview | bytes | bytearray, captured_ns: int) -> int:
        if len(frame_bytes) != self.frame_bytes:
            raise ValueError(f"frame bytes={len(frame_bytes)} expected={self.frame_bytes}")
        next_slot = 1 - self.slot
        offset = HEADER_SIZE + (next_slot * self.frame_bytes)
        self.mm[offset : offset + self.frame_bytes] = frame_bytes
        self.slot = next_slot
        self.seq += 2
        self._publish_header(captured_ns=captured_ns)
        return self.seq

    def close(self) -> None:
        try:
            self.mm.flush()
        except Exception:
            pass
        try:
            self.mm.close()
        finally:
            os.close(self.fd)


class LatestFrameMmapReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.fd = os.open(self.path, os.O_RDONLY)
        self.size = os.fstat(self.fd).st_size
        self.mm = mmap.mmap(self.fd, self.size, access=mmap.ACCESS_READ)

    def _header(self):
        row = _HEADER.unpack_from(self.mm, 0)
        magic, version, seq, slot, width, height, stride, frame_bytes, captured_ns = row
        if magic != MAGIC or version != VERSION:
            raise RuntimeError(f"invalid camera SHM header magic={magic!r} version={version}")
        return seq, slot, width, height, stride, frame_bytes, captured_ns

    def latest(self, retries: int = 4) -> FrameSnapshot | None:
        for _ in range(max(1, retries)):
            first = self._header()
            seq, slot, width, height, stride, frame_bytes, captured_ns = first
            if seq <= 0:
                return None
            offset = HEADER_SIZE + (slot * frame_bytes)
            if offset + frame_bytes > self.size:
                raise RuntimeError("camera SHM slot exceeds mapping")
            data = bytes(self.mm[offset : offset + frame_bytes])
            second = self._header()
            if first == second:
                return FrameSnapshot(
                    seq=int(seq),
                    slot=int(slot),
                    width=int(width),
                    height=int(height),
                    stride=int(stride),
                    frame_bytes=int(frame_bytes),
                    captured_ns=int(captured_ns),
                    data=data,
                )
        return None

    def close(self) -> None:
        try:
            self.mm.close()
        finally:
            os.close(self.fd)


class FrameDemandOwner:
    """Camera-side owner of a tiny per-camera ML demand counter.

    The ML process increments request_seq. The camera process only reads it from
    its streaming probe and keeps the served counter in-process. No locks, socket
    queues or historical frame backlog are introduced into the camera data-plane.
    """

    def __init__(self, camera_id: str) -> None:
        safe = _safe_camera_name(camera_id)
        self.path = _shm_root() / f"{safe}.request"
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(self.fd, REQUEST_SIZE)
        self.mm = mmap.mmap(self.fd, REQUEST_SIZE, access=mmap.ACCESS_WRITE)
        _REQUEST_HEADER.pack_into(
            self.mm, 0, REQUEST_MAGIC, REQUEST_VERSION, 0, 0
        )
        if _REQUEST_HEADER.size < REQUEST_SIZE:
            self.mm[_REQUEST_HEADER.size:REQUEST_SIZE] = b"\0" * (
                REQUEST_SIZE - _REQUEST_HEADER.size
            )

    def requested_seq(self) -> int:
        magic, version, _reserved, request_seq = _REQUEST_HEADER.unpack_from(self.mm, 0)
        if magic != REQUEST_MAGIC or version != REQUEST_VERSION:
            raise RuntimeError(
                f"invalid demand header magic={magic!r} version={version}"
            )
        return int(request_seq)

    def close(self) -> None:
        try:
            self.mm.close()
        finally:
            os.close(self.fd)


class FrameDemandClient:
    """ML-side client that requests exactly one future frame at a time."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.fd = os.open(self.path, os.O_RDWR)
        self.size = os.fstat(self.fd).st_size
        if self.size < REQUEST_SIZE:
            os.close(self.fd)
            raise RuntimeError(f"demand file too small: {self.path}")
        self.mm = mmap.mmap(self.fd, REQUEST_SIZE, access=mmap.ACCESS_WRITE)
        magic, version, _reserved, request_seq = _REQUEST_HEADER.unpack_from(self.mm, 0)
        if magic != REQUEST_MAGIC or version != REQUEST_VERSION:
            self.mm.close()
            os.close(self.fd)
            raise RuntimeError(
                f"invalid demand header magic={magic!r} version={version}"
            )
        self.seq = int(request_seq)

    def request(self) -> int:
        # One ML scheduler owns this client, so request_seq has a single writer.
        self.seq += 1
        _REQUEST_SEQ.pack_into(self.mm, _REQUEST_SEQ_OFFSET, int(self.seq))
        return self.seq

    def close(self) -> None:
        try:
            self.mm.close()
        finally:
            os.close(self.fd)
