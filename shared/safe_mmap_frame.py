from __future__ import annotations

import errno
import mmap
import os
import threading
import time
from pathlib import Path

from .mmap_frame import (
    HEADER_SIZE,
    MAGIC,
    META_OFFSET,
    SEQ_OFFSET,
    MmapFrameWriter,
    _META,
    _U64,
    frame_path,
)


class SigbusSafeMmapFrameWriter(MmapFrameWriter):
    """Mmap writer that never truncates a file currently mapped by the UI.

    Linux can deliver SIGBUS when a process touches pages beyond the current
    end of a mapped file, and POSIX also permits SIGBUS when backing storage
    cannot satisfy a mapped write. The original writer opened the public
    target with O_TRUNC, which made backend restarts unsafe while Qt still had
    the previous inode mapped.

    This writer instead creates and fully preallocates a private temporary
    inode, maps and initializes it, then atomically publishes it with
    os.replace(). Existing readers keep a valid mapping to the old inode until
    they notice the path inode changed and reattach.
    """

    def __init__(self, camera_id: str, max_width: int, max_height: int, channels: int = 3):
        self.camera_id = str(camera_id)
        self.max_width = max(1, int(max_width))
        self.max_height = max(1, int(max_height))
        self.channels = max(1, int(channels))
        self.slot_bytes = self.max_width * self.max_height * self.channels
        self.total_bytes = HEADER_SIZE + 2 * self.slot_bytes
        self.path = frame_path(self.camera_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        token = f"{os.getpid()}.{threading.get_ident()}.{time.monotonic_ns()}"
        temp_path = self.path.with_name(f".{self.path.name}.{token}.tmp")
        fd = -1
        mapped = None
        try:
            fd = os.open(str(temp_path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            self._reserve_backing_store(fd, self.total_bytes)
            os.ftruncate(fd, self.total_bytes)
            mapped = mmap.mmap(fd, self.total_bytes, access=mmap.ACCESS_WRITE)

            mapped[0:8] = MAGIC
            mapped[SEQ_OFFSET:SEQ_OFFSET + 8] = _U64.pack(0)
            mapped[META_OFFSET:HEADER_SIZE] = _META.pack(
                0, 0, 0, self.channels, 0, 0, 0, 0
            )

            # Publish only after the new inode is complete. os.replace() does
            # not shrink or mutate the old inode held by existing readers.
            os.replace(str(temp_path), str(self.path))
        except Exception:
            if mapped is not None:
                try:
                    mapped.close()
                except Exception:
                    pass
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            raise

        self._fd = fd
        self._map = mapped
        self._inode = int(os.fstat(fd).st_ino)
        self._sequence = 0
        self._active_slot = 0
        self._closed = False

    @staticmethod
    def _reserve_backing_store(fd: int, length: int) -> None:
        """Reserve every page before mmap so ENOSPC is a Python error, not SIGBUS."""
        length = max(1, int(length))
        allocator = getattr(os, "posix_fallocate", None)
        if allocator is not None:
            allocator(fd, 0, length)
            return

        # Conservative fallback for platforms without posix_fallocate.
        stats = os.fstatvfs(fd)
        available = int(stats.f_bavail) * int(stats.f_frsize)
        if available < length:
            raise OSError(errno.ENOSPC, "insufficient backing storage for mmap frame")
        os.ftruncate(fd, length)

    def close(self, unlink: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        mapped, fd = self._map, self._fd
        self._map = None
        self._fd = -1

        if mapped is not None:
            try:
                mapped.close()
            except (BufferError, OSError):
                pass
        if fd is not None and fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass

        if not unlink:
            return

        # Never unlink a newer writer's atomically replaced target.
        try:
            current = os.stat(self.path)
            if int(current.st_ino) != int(self._inode):
                return
            self.path.unlink()
        except FileNotFoundError:
            pass
