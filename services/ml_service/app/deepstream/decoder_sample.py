from __future__ import annotations

from pathlib import Path


class BoundedDecoderSample:
    """Opt-in, bounded elementary-stream sample for offline codec inspection."""

    def __init__(self, directory: str | Path, camera_id: str, codec: str, max_bytes: int = 2 * 1024 * 1024):
        self.directory = Path(directory)
        self.camera_id = camera_id.lower().replace("-", "_")
        self.codec = codec.lower()
        if self.codec not in {"h264", "h265"}:
            raise ValueError(f"unsupported decoder sample codec: {codec}")
        self.max_bytes = max(0, int(max_bytes))
        self.bytes_written = 0
        self._handle = None
        self.path = self.directory / f"{self.camera_id}.{self.codec}"

    def append(self, payload) -> int:
        remaining = self.max_bytes - self.bytes_written
        if remaining <= 0:
            return 0
        chunk = memoryview(payload)[:remaining]
        if not chunk:
            return 0
        if self._handle is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("wb", buffering=0)
        written = self._handle.write(chunk)
        self.bytes_written += written
        return written

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None
