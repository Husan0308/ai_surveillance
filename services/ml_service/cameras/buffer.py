"""Thread-safe capacity-one latest-frame buffer."""
from __future__ import annotations

import threading
from typing import Callable, Optional

from .frame import FramePacket


class LatestFrameBuffer:
    """A mailbox: every put replaces an unconsumed packet instead of queueing."""
    maxsize = 1

    def __init__(self, on_available: Optional[Callable[[], None]] = None) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._packet: FramePacket | None = None
        self._closed = False
        self._on_available = on_available
        self.dropped_old = 0

    def put(self, packet: FramePacket) -> None:
        with self._condition:
            if self._closed:
                return
            if self._packet is not None:
                self.dropped_old += 1
            self._packet = packet
            self._condition.notify_all()
        if self._on_available is not None:
            self._on_available()

    def take(self) -> FramePacket | None:
        with self._condition:
            packet, self._packet = self._packet, None
            return packet

    def peek(self) -> FramePacket | None:
        with self._condition:
            return self._packet

    def wait_and_take(self, timeout: float | None = None) -> FramePacket | None:
        with self._condition:
            self._condition.wait_for(lambda: self._packet is not None or self._closed, timeout)
            packet, self._packet = self._packet, None
            return packet

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._packet = None
            self._condition.notify_all()

    def __len__(self) -> int:
        with self._condition:
            return int(self._packet is not None)
