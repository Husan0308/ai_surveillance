from __future__ import annotations

import fcntl
import mmap
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"V11UI01\0"
VERSION = 1
HEADER_SIZE = 64
HEADER = struct.Struct("<8sIQQIIIIII")
DEFAULT_PATH = "/dev/shm/v11_ui_preview_cam01_v1.bin"


@dataclass(frozen=True)
class PreviewFrame:
    sequence: int
    timestamp_ns: int
    width: int
    height: int
    stride: int
    object_count: int
    fps: float
    payload: bytes


class PreviewFrameWriter:
    def __init__(self, path: str = DEFAULT_PATH, width: int = 640, height: int = 360, stride: int | None = None):
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.stride = int(stride or self.width * 4)
        self.payload_size = self.stride * self.height
        self.sequence = 0
        self._last_publish = 0.0
        self._fps_ema = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        os.ftruncate(self.fd, HEADER_SIZE + self.payload_size)
        self.mm = mmap.mmap(self.fd, HEADER_SIZE + self.payload_size, access=mmap.ACCESS_WRITE)
        self._write_header(timestamp_ns=0, object_count=0, fps_milli=0)

    def _write_header(self, timestamp_ns: int, object_count: int, fps_milli: int) -> None:
        packed = HEADER.pack(
            MAGIC, VERSION, self.sequence, int(timestamp_ns), self.width, self.height,
            self.stride, self.payload_size, max(0, int(object_count)), max(0, int(fps_milli)),
        )
        self.mm.seek(0)
        self.mm.write(packed)
        if len(packed) < HEADER_SIZE:
            self.mm.write(b"\0" * (HEADER_SIZE - len(packed)))

    def publish(self, payload, object_count: int = 0, timestamp_ns: int | None = None) -> int:
        view = memoryview(payload)
        if view.nbytes < self.payload_size:
            raise ValueError(f"preview payload too small {view.nbytes}<{self.payload_size}")
        now = time.monotonic()
        if self._last_publish > 0.0:
            instant = 1.0 / max(1e-6, now - self._last_publish)
            self._fps_ema = instant if self._fps_ema <= 0.0 else (self._fps_ema * 0.85 + instant * 0.15)
        self._last_publish = now
        self.sequence += 1
        stamp = int(timestamp_ns if timestamp_ns is not None else time.monotonic_ns())
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        try:
            self.mm.seek(HEADER_SIZE)
            self.mm.write(view[: self.payload_size])
            self._write_header(stamp, object_count, int(round(self._fps_ema * 1000.0)))
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        return self.sequence

    def close(self, unlink: bool = True) -> None:
        try:
            self.mm.close()
        finally:
            os.close(self.fd)
        if unlink:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


class PreviewFrameReader:
    def __init__(self, path: str = DEFAULT_PATH):
        self.path = Path(path)
        self.fd: int | None = None
        self.mm: mmap.mmap | None = None
        self.size = 0

    def _ensure_open(self) -> bool:
        if self.mm is not None and self.fd is not None:
            try:
                current = os.stat(self.path)
                opened = os.fstat(self.fd)
                if (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino):
                    return True
            except OSError:
                pass
            self.close()
        try:
            fd = os.open(self.path, os.O_RDONLY)
            size = os.fstat(fd).st_size
            if size < HEADER_SIZE:
                os.close(fd)
                return False
            mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        except (FileNotFoundError, OSError, ValueError):
            return False
        self.fd, self.mm, self.size = fd, mm, size
        return True

    def read_latest(self, max_age_sec: float = 1.20) -> PreviewFrame | None:
        if not self._ensure_open():
            return None
        assert self.fd is not None and self.mm is not None
        locked = False
        try:
            fcntl.flock(self.fd, fcntl.LOCK_SH)
            locked = True
            self.mm.seek(0)
            raw = self.mm.read(HEADER_SIZE)
            fields = HEADER.unpack(raw[: HEADER.size])
            magic, version, sequence, timestamp_ns, width, height, stride, payload_size, object_count, fps_milli = fields
            if magic != MAGIC or version != VERSION or sequence <= 0:
                return None
            if HEADER_SIZE + payload_size > self.size:
                return None
            if timestamp_ns <= 0 or (time.monotonic_ns() - timestamp_ns) > int(max_age_sec * 1e9):
                return None
            self.mm.seek(HEADER_SIZE)
            payload = self.mm.read(payload_size)
            return PreviewFrame(
                sequence=int(sequence), timestamp_ns=int(timestamp_ns), width=int(width), height=int(height),
                stride=int(stride), object_count=int(object_count), fps=float(fps_milli) / 1000.0,
                payload=payload,
            )
        except (BufferError, OSError, ValueError, struct.error):
            self.close()
            return None
        finally:
            if locked and self.fd is not None:
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
                except OSError:
                    pass

    def close(self) -> None:
        if self.mm is not None:
            try:
                self.mm.close()
            except Exception:
                pass
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
        self.mm = None
        self.fd = None
        self.size = 0
