from __future__ import annotations

import mmap
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"AISF"
VERSION = 1
HEADER_STRUCT = struct.Struct("<4sIQQIIII")
HEADER_SIZE = 64


def _safe_name(camera_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", camera_id.strip())
    if not value:
        raise ValueError("camera_id must not be empty")
    return value


def frame_path(directory: str | Path, camera_id: str) -> Path:
    return Path(directory) / f"{_safe_name(camera_id)}.frame"


@dataclass(frozen=True)
class FramePacket:
    sequence: int
    timestamp_ns: int
    width: int
    height: int
    stride: int
    data: bytes


class LatestFrameWriter:
    """Single-writer latest-frame shared-memory file."""

    def __init__(self, directory: str | Path, camera_id: str, max_payload_bytes: int) -> None:
        if max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")
        self.path = frame_path(directory, camera_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_payload_bytes = int(max_payload_bytes)
        self.total_size = HEADER_SIZE + self.max_payload_bytes
        self._sequence = 0
        self._closed = False
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, self.total_size)
            self._mmap = mmap.mmap(fd, self.total_size, access=mmap.ACCESS_WRITE)
        finally:
            os.close(fd)
        self._write_header(sequence=0, timestamp_ns=0, width=0, height=0, stride=0, payload_size=0)

    def _write_header(self, *, sequence: int, timestamp_ns: int, width: int, height: int, stride: int, payload_size: int) -> None:
        HEADER_STRUCT.pack_into(self._mmap, 0, MAGIC, VERSION, int(sequence), int(timestamp_ns), int(width), int(height), int(stride), int(payload_size))

    def publish(self, data: bytes | bytearray | memoryview, *, timestamp_ns: int, width: int, height: int, stride: int) -> int:
        if self._closed:
            raise RuntimeError("frame writer is closed")
        if width <= 0 or height <= 0 or stride <= 0:
            raise ValueError("invalid frame geometry")
        view = memoryview(data).cast("B")
        payload_size = len(view)
        if payload_size <= 0:
            raise ValueError("empty frame payload")
        if payload_size > self.max_payload_bytes:
            raise ValueError(f"frame payload {payload_size} exceeds shared buffer capacity {self.max_payload_bytes}")
        next_even = self._sequence + 2
        in_progress = next_even - 1
        self._write_header(sequence=in_progress, timestamp_ns=timestamp_ns, width=width, height=height, stride=stride, payload_size=payload_size)
        self._mmap[HEADER_SIZE:HEADER_SIZE + payload_size] = view
        self._write_header(sequence=next_even, timestamp_ns=timestamp_ns, width=width, height=height, stride=stride, payload_size=payload_size)
        self._sequence = next_even
        return next_even

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._mmap.close()


class LatestFrameReader:
    """Lock-free reader for LatestFrameWriter files."""

    def __init__(self, directory: str | Path, camera_id: str) -> None:
        self.path = frame_path(directory, camera_id)
        self._mmap: mmap.mmap | None = None
        self._size = 0

    def _close_mapping(self) -> None:
        mapping = self._mmap
        self._mmap = None
        self._size = 0
        if mapping is not None:
            mapping.close()

    def _ensure_open(self) -> bool:
        if self._mmap is not None:
            return True
        try:
            fd = os.open(self.path, os.O_RDONLY)
        except FileNotFoundError:
            return False
        try:
            size = os.fstat(fd).st_size
            if size < HEADER_SIZE:
                return False
            self._mmap = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
            self._size = size
            return True
        finally:
            os.close(fd)

    def _read_header(self) -> tuple[bytes, int, int, int, int, int, int, int]:
        if self._mmap is None:
            raise RuntimeError("frame buffer is not open")
        return HEADER_STRUCT.unpack_from(self._mmap, 0)

    def read_latest(self, last_sequence: int | None = None, *, retries: int = 3) -> FramePacket | None:
        if not self._ensure_open():
            return None
        for _ in range(max(1, retries)):
            try:
                magic, version, sequence_before, timestamp_ns, width, height, stride, payload_size = self._read_header()
            except (ValueError, OSError):
                self._close_mapping()
                return None
            if magic != MAGIC or version != VERSION:
                return None
            if sequence_before == 0 or sequence_before & 1:
                continue
            if last_sequence is not None and sequence_before == last_sequence:
                return None
            if width <= 0 or height <= 0 or stride <= 0 or payload_size <= 0 or HEADER_SIZE + payload_size > self._size:
                return None
            try:
                payload = bytes(self._mmap[HEADER_SIZE:HEADER_SIZE + payload_size])
                magic_after, version_after, sequence_after, _timestamp_after, _width_after, _height_after, _stride_after, _payload_after = self._read_header()
            except (ValueError, OSError):
                self._close_mapping()
                return None
            if magic_after == MAGIC and version_after == VERSION and sequence_before == sequence_after and not (sequence_after & 1):
                return FramePacket(sequence=sequence_after, timestamp_ns=timestamp_ns, width=width, height=height, stride=stride, data=payload)
        return None

    def close(self) -> None:
        self._close_mapping()
