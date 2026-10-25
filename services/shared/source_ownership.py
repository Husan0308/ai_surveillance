"""Cooperative per-camera ownership, acquired before opening any RTSP session.

Keep lock files: unlinking a flock inode would allow a second owner to lock a
different inode. The directory must be shared with containerized owners.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import re


class SourceOwnership:
    def __init__(self, cameras, directory: Path):
        self.cameras = tuple(sorted(cameras))
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError("duplicate camera selection")
        if not self.cameras or any(not re.fullmatch(r"CAM-\d{2}", c) for c in self.cameras):
            raise ValueError("ownership requires CAM-XX camera IDs")
        self.directory = Path(directory)
        self.fds: list[int] = []

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            for camera in self.cameras:
                fd = os.open(self.directory / f"{camera}.lock", os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    os.close(fd)
                    raise RuntimeError(f"{camera}: another camera owner holds the source lock") from exc
                self.fds.append(fd)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for fd in reversed(self.fds):
            os.close(fd)
        self.fds.clear()
