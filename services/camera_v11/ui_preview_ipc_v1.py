from __future__ import annotations

import fcntl
import json
import mmap
import os
import socket
import stat
import struct
import time
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"V11UI01\0"
VERSION = 4
HEADER_SIZE_V1 = 64
HEADER_SIZE_V3 = 96
HEADER_SIZE = 104
HEADER_SIZE_V5 = 65536
HEADER_V5_EXT = struct.Struct("<QQQII")
HEADER_V1 = struct.Struct("<8sIQQIIIIII")
# Version 2 used eight bytes that were padding in the 64-byte v1 header.
# Version 3 has a 96-byte header carrying decoder input/output timing, PTS and
# source frame number. Readers remain compatible with native v1 publishers.
HEADER_V2 = struct.Struct("<8sIQQIIIIIIQ")
HEADER_V3 = struct.Struct("<8sIQQIIIIIIQQQII")
HEADER_V4 = struct.Struct("<8sIQQIIIIIIQQQQII")
DEFAULT_PATH = "/dev/shm/v11_ui_preview_cam01_v1.bin"
DEFAULT_NOTIFY_PATH = "/dev/shm/v11_ui_preview_notify_v1.sock"


class PreviewFrameNotifyReceiver:
    """Bounded event wakeup for newly published shared-memory camera frames.

    The normal timer remains a stale-state safety net. Native publication sends
    a two-byte Unix datagram (tile/fullscreen, source index) so Qt can render
    only the changed camera immediately instead of waiting for the next tick.
    """

    def __init__(self, path: str | Path = DEFAULT_NOTIFY_PATH) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self._lock_fd)
            self._lock_fd = -1
            raise RuntimeError("another preview UI owns the notification socket") from exc

        self.socket: socket.socket | None = socket.socket(
            socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_NONBLOCK
        )
        try:
            if self.path.exists():
                if not stat.S_ISSOCK(self.path.lstat().st_mode):
                    raise RuntimeError(f"refusing to replace non-socket preview path: {self.path}")
                self.path.unlink()
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            self.socket.bind(str(self.path))
        except Exception:
            self.socket.close()
            self.socket = None
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = -1
            raise

    def fileno(self) -> int:
        return self.socket.fileno()

    def drain(self) -> set[tuple[str, int]]:
        """Drain queued events, coalescing them to one refresh per source."""
        pending: set[tuple[str, int]] = set()
        while True:
            try:
                payload = self.socket.recv(16)
            except BlockingIOError:
                break
            if len(payload) != 2 or payload[0:1] not in (b"T", b"F") or payload[1] >= 6:
                continue
            pending.add(("tile" if payload[0:1] == b"T" else "fullscreen", payload[1]))
        return pending

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        try:
            if self.path.exists() and stat.S_ISSOCK(self.path.lstat().st_mode):
                self.path.unlink()
        except OSError:
            pass
        if self._lock_fd >= 0:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = -1


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
    decoder_reference_ns: int = 0
    decoder_out_ns: int = 0
    pts_ns: int = 0
    decoder_dts_ns: int = 0
    source_frame_num: int = 0
    tracker_frame_num: int = 0
    preview_prepare_start_ns: int = 0
    preview_prepare_complete_ns: int = 0
    tracker_ready_ns: int = 0
    tracker_snapshot: dict | None = None


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

    def _write_header(self, timestamp_ns: int, object_count: int, fps_milli: int,
                      decoder_reference_ns: int = 0, decoder_out_ns: int = 0,
                      pts_ns: int = 0, decoder_dts_ns: int = 0,
                      source_frame_num: int = 0) -> None:
        packed = HEADER_V4.pack(
            MAGIC, VERSION, self.sequence, int(timestamp_ns), self.width, self.height,
            self.stride, self.payload_size, max(0, int(object_count)), max(0, int(fps_milli)),
            max(0, int(decoder_reference_ns)), max(0, int(decoder_out_ns)),
            max(0, int(pts_ns)), max(0, int(decoder_dts_ns)),
            max(0, int(source_frame_num)), 0,
        )
        self.mm.seek(0)
        self.mm.write(packed)
        if len(packed) < HEADER_SIZE:
            self.mm.write(b"\0" * (HEADER_SIZE - len(packed)))

    def publish(self, payload, object_count: int = 0, timestamp_ns: int | None = None,
                fps: float | None = None, decoder_reference_ns: int = 0,
                decoder_out_ns: int = 0, pts_ns: int = 0,
                decoder_dts_ns: int = 0,
                source_frame_num: int = 0) -> int:
        view = memoryview(payload)
        if view.nbytes < self.payload_size:
            raise ValueError(f"preview payload too small {view.nbytes}<{self.payload_size}")
        now = time.monotonic()
        if self._last_publish > 0.0:
            instant = 1.0 / max(1e-6, now - self._last_publish)
            self._fps_ema = instant if self._fps_ema <= 0.0 else (self._fps_ema * 0.85 + instant * 0.15)
        if fps is not None and fps >= 0.0:
            self._fps_ema = float(fps)
        self._last_publish = now
        self.sequence += 1
        stamp = int(timestamp_ns if timestamp_ns is not None else time.monotonic_ns())
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        try:
            self.mm.seek(HEADER_SIZE)
            self.mm.write(view[: self.payload_size])
            self._write_header(
                stamp, object_count, int(round(self._fps_ema * 1000.0)),
                decoder_reference_ns, decoder_out_ns, pts_ns, decoder_dts_ns,
                source_frame_num,
            )
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
                if (current.st_dev, current.st_ino, current.st_size) == (
                    opened.st_dev, opened.st_ino, self.size
                ):
                    return True
            except OSError:
                pass
            # DeepStream unlinks and recreates the preview on restart. The
            # old mmap remains readable but can only return a stale frame.
            self.close()
        try:
            fd = os.open(self.path, os.O_RDONLY)
            size = os.fstat(fd).st_size
            if size < HEADER_SIZE_V1:
                os.close(fd)
                return False
            mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        except (FileNotFoundError, OSError, ValueError):
            return False
        self.fd, self.mm, self.size = fd, mm, size
        return True

    def read_latest(
        self,
        max_age_sec: float = 1.20,
        after_sequence: int | None = None,
        metadata_only: bool = False,
    ) -> PreviewFrame | None:
        if not self._ensure_open():
            return None
        assert self.fd is not None and self.mm is not None
        locked = False
        try:
            fcntl.flock(self.fd, fcntl.LOCK_SH)
            locked = True
            self.mm.seek(0)
            raw = self.mm.read(min(HEADER_SIZE_V5, self.size))
            prefix = HEADER_V1.unpack(raw[: HEADER_V1.size])
            magic, version, sequence, timestamp_ns, width, height, stride, payload_size, object_count, fps_milli = prefix
            if magic != MAGIC or version not in (1, 2, 3, 4, 5) or sequence <= 0:
                return None
            # The camera wall polls every GUI tick. Avoid copying megabytes of
            # unchanged pixels into Python bytes while the shared slot still
            # contains the frame we already painted. A smaller sequence is a
            # writer restart (its mmap inode was recreated), so accept it.
            if after_sequence is not None and sequence == int(after_sequence):
                return None
            header_size = HEADER_SIZE_V1
            decoder_reference_ns = 0
            decoder_out_ns = 0
            pts_ns = 0
            decoder_dts_ns = 0
            source_frame_num = 0
            tracker_frame_num = 0
            preview_prepare_start_ns = 0
            preview_prepare_complete_ns = 0
            tracker_ready_ns = 0
            tracker_snapshot = None
            if version == 2:
                decoder_out_ns = int(HEADER_V2.unpack(raw[: HEADER_V2.size])[-1])
            elif version == 3:
                header_size = HEADER_SIZE_V3
                unpacked = HEADER_V3.unpack(raw[: HEADER_V3.size])
                decoder_reference_ns = int(unpacked[10])
                decoder_out_ns = int(unpacked[11])
                pts_ns = int(unpacked[12])
                source_frame_num = int(unpacked[13])
            elif version >= 4:
                header_size = HEADER_SIZE_V5 if version == 5 else HEADER_SIZE
                unpacked_v4 = HEADER_V4.unpack(raw[: HEADER_V4.size])
                decoder_reference_ns = int(unpacked_v4[10])
                decoder_out_ns = int(unpacked_v4[11])
                pts_ns = int(unpacked_v4[12])
                decoder_dts_ns = int(unpacked_v4[13])
                source_frame_num = int(unpacked_v4[14])
                if version == 5:
                    ext = HEADER_V5_EXT.unpack(raw[92:92 + HEADER_V5_EXT.size])
                    preview_prepare_start_ns = int(ext[0])
                    preview_prepare_complete_ns = int(ext[1])
                    tracker_ready_ns = int(ext[2])
                    tracker_frame_num = int(ext[3])
                    metadata_size = int(ext[4])
                    if metadata_size > HEADER_SIZE_V5 - 128:
                        return None
                    if metadata_size:
                        tracker_snapshot = json.loads(raw[128:128 + metadata_size].decode("utf-8"))
                        if (not isinstance(tracker_snapshot, dict)
                                or int(tracker_snapshot.get("pts_ns", -1)) != pts_ns
                                or int(tracker_snapshot.get("frame_num", -1)) != tracker_frame_num
                                or int(tracker_snapshot.get("source_frame_num", -1)) != source_frame_num):
                            return None
            if (width <= 0 or height <= 0 or stride < width * 4
                    or payload_size != stride * height
                    or header_size + payload_size > self.size):
                return None
            age_ns = time.monotonic_ns() - timestamp_ns
            if timestamp_ns <= 0 or age_ns < 0 or age_ns > int(max_age_sec * 1e9):
                return None
            self.mm.seek(header_size)
            # Health observers need sequence/time evidence, not another full
            # pixel copy. The UI's default read path is unchanged.
            payload = b"" if metadata_only else self.mm.read(payload_size)
            return PreviewFrame(
                sequence=int(sequence), timestamp_ns=int(timestamp_ns), width=int(width), height=int(height),
                stride=int(stride), object_count=int(object_count), fps=float(fps_milli) / 1000.0,
                payload=payload, decoder_reference_ns=decoder_reference_ns,
                decoder_out_ns=decoder_out_ns, pts_ns=pts_ns,
                decoder_dts_ns=decoder_dts_ns,
                source_frame_num=source_frame_num,
                tracker_frame_num=tracker_frame_num,
                preview_prepare_start_ns=preview_prepare_start_ns,
                preview_prepare_complete_ns=preview_prepare_complete_ns,
                tracker_ready_ns=tracker_ready_ns,
                tracker_snapshot=tracker_snapshot,
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
