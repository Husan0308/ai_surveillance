from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path

from services.ml_service.core_v1.latest_frame import Frame, LatestFrameStore


ROOT = Path(__file__).resolve().parents[1]


class EventDrivenPublisherTests(unittest.TestCase):
    def test_latest_frame_store_wakes_on_new_frame(self):
        store = LatestFrameStore()
        result = {}

        def waiter():
            started = time.monotonic()
            frame, version = store.wait_newer(0, timeout=0.5)
            result["elapsed"] = time.monotonic() - started
            result["frame"] = frame
            result["version"] = version

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.03)
        store.put(Frame("CAM-01", 1, time.time(), time.monotonic(), object(), 736, 416))
        thread.join(1.0)

        self.assertFalse(thread.is_alive())
        self.assertIsNotNone(result.get("frame"))
        self.assertEqual(result.get("version"), 1)
        self.assertLess(result.get("elapsed", 1.0), 0.20)

    def test_latest_store_remains_single_slot(self):
        store = LatestFrameStore()
        first = Frame("CAM-01", 1, 0.0, 1.0, "first", 736, 416)
        second = Frame("CAM-01", 2, 0.0, 2.0, "second", 736, 416)
        store.put(first)
        store.put(second)
        frame, version = store.get()
        self.assertEqual(version, 2)
        self.assertEqual(frame.frame_id, 2)
        self.assertEqual(frame.image, "second")
        self.assertEqual(store.replaced, 1)

    def test_tracking_publisher_uses_event_driven_base(self):
        source = (ROOT / "services/ml_service/core_v1/tracking_publisher.py").read_text(encoding="utf-8")
        event_source = (ROOT / "services/ml_service/core_v1/event_publisher.py").read_text(encoding="utf-8")
        self.assertIn("EventDrivenJpegPublisher", source)
        self.assertIn("class TrackingJpegPublisher(EventDrivenJpegPublisher)", source)
        self.assertIn("self.store.wait_newer", event_source)
        self.assertIn('"publisher_mode": "event-driven-latest-only"', event_source)
        self.assertNotIn("next_at += self.interval", event_source)


if __name__ == "__main__":
    unittest.main()
