import threading
import time
import unittest
from services.ml_service.cameras.buffer import LatestFrameBuffer
from services.ml_service.cameras.frame import FramePacket
from services.ml_service.pipeline.scheduler import BatchScheduler

class CameraBatchingTests(unittest.TestCase):
    def packet(self, camera, frame_id, timestamp=None):
        stamp = timestamp or time.time()
        return FramePacket(camera, frame_id, stamp, stamp, object(), 640, 360)

    def test_latest_frame_replaces_old(self):
        buffer = LatestFrameBuffer()
        buffer.put(self.packet("CAM-01", 1)); buffer.put(self.packet("CAM-01", 2))
        self.assertEqual(buffer.take().frame_id, 2)
        self.assertEqual(buffer.dropped_old, 1); self.assertIsNone(buffer.take())

    def test_partial_batch_does_not_wait_for_starved_camera(self):
        delivered, ready = [], threading.Event()
        scheduler = BatchScheduler(250, lambda batch: (delivered.append(batch), ready.set()), batch_collect_window_ms=10)
        buffers = {camera: LatestFrameBuffer(scheduler.notify_frame_available) for camera in ("CAM-01", "CAM-02", "CAM-03")}
        for camera, buffer in buffers.items(): scheduler.register_camera(camera, buffer)
        scheduler.start()
        buffers["CAM-01"].put(self.packet("CAM-01", 1)); buffers["CAM-03"].put(self.packet("CAM-03", 1))
        self.assertTrue(ready.wait(1)); scheduler.stop(); scheduler.join(1)
        self.assertEqual(delivered[0].camera_ids, ("CAM-01", "CAM-03"))

    def test_stale_and_duplicate_frames_are_not_scheduled(self):
        delivered = []
        scheduler = BatchScheduler(20, delivered.append, batch_collect_window_ms=1)
        buffer = LatestFrameBuffer(scheduler.notify_frame_available); scheduler.register_camera("CAM-01", buffer); scheduler.start()
        buffer.put(self.packet("CAM-01", 1, time.time() - 1)); time.sleep(.05)
        buffer.put(self.packet("CAM-01", 2)); time.sleep(.05)
        buffer.put(self.packet("CAM-01", 2)); time.sleep(.05)
        scheduler.stop(); scheduler.join(1)
        metrics = scheduler.snapshot_metrics({"CAM-01": {"online": True}})
        self.assertEqual(len(delivered), 1); self.assertEqual(metrics["stale_drops"], 1)
        self.assertEqual(metrics["cameras"]["CAM-01"]["duplicate_count"], 1)

if __name__ == "__main__": unittest.main()
