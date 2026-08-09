import time
import unittest
import numpy as np
from services.ml_service.cameras.buffer import LatestFrameBuffer
from services.ml_service.cameras.frame import FramePacket
from services.ml_service.detection.person_detector import PersonDetector,filter_end2end_predictions
from services.ml_service.pipeline.batch import BatchOutput
from services.ml_service.pipeline.scheduler import BatchScheduler

CONFIG = {"ai": {"max_frame_age_ms": 100, "detector": {"imgsz": [100, 100],
    "min_box_width": 1, "min_box_height": 1, "low_conf_size_threshold": 0}}}

class FakeBackend:
    coordinates_original = False
    def __init__(self, rows=None): self.rows, self.calls, self.batch_sizes = rows, 0, []
    def infer(self, prepared):
        self.calls += 1; self.batch_sizes.append(len(prepared.batch.frames))
        rows = self.rows if self.rows is not None else np.array([[25, 30, 75, 70, .9, 0]], np.float32)
        return [rows.copy() for _ in prepared.batch.frames], {"gpu_inference_ms": 2.0}

def packet(camera, frame_id, timestamp=None):
    stamp = timestamp or time.time(); frame = np.zeros((100, 200, 3), np.uint8)
    return FramePacket(camera, frame_id, stamp, stamp, frame, 200, 100)

class PersonDetectorTests(unittest.TestCase):
    def test_end2end_batch_filter_matches_threshold_class_and_order(self):
        raw=np.array([[[1,2,3,4,.9,0],[2,3,4,5,.8,1],[3,4,5,6,.05,0]],[[5,6,7,8,.7,0],[6,7,8,9,.6,0],[7,8,9,10,.5,0]]],np.float32)
        output=filter_end2end_predictions(raw,.05,[0],2)
        np.testing.assert_array_equal(output[0],raw[0,:1]);np.testing.assert_array_equal(output[1],raw[1,:2])
        output=filter_end2end_predictions(raw,.05,[0],1)
        np.testing.assert_array_equal(output[0],raw[0,:1]);np.testing.assert_array_equal(output[1],raw[1,:1])
    def test_dynamic_batches_and_identity(self):
        for size in range(1, 7):
            with self.subTest(size=size):
                backend = FakeBackend(); detector = PersonDetector(CONFIG, backend)
                batch = BatchOutput(size, time.time(), tuple(packet(f"CAM-{i:02d}", i) for i in range(1, size + 1)))
                result = detector.process_batch(batch)
                self.assertEqual(backend.batch_sizes, [size]); self.assertEqual(len(result.results), size)
                self.assertEqual([(r.camera_id, r.frame_id) for r in result.results],
                                 [(f"CAM-{i:02d}", i) for i in range(1, size + 1)])

    def test_original_coordinate_conversion_and_clipping(self):
        backend = FakeBackend(np.array([[-10, 20, 110, 80, .8, 0]], np.float32))
        result = PersonDetector(CONFIG, backend).process_batch(BatchOutput(1, time.time(), (packet("CAM-01", 1),)))
        self.assertEqual(result.results[0].detections[0].bbox_xyxy, (0.0, 0.0, 200.0, 100.0))

    def test_person_filter_and_empty_detections(self):
        rows = np.array([[10, 10, 50, 50, .9, 2]], np.float32)
        result = PersonDetector(CONFIG, FakeBackend(rows)).process_batch(BatchOutput(1, time.time(), (packet("CAM-01", 1),)))
        self.assertEqual(result.results[0].detections, ())
        result = PersonDetector(CONFIG, FakeBackend(np.empty((0, 6), np.float32))).process_batch(BatchOutput(2, time.time(), (packet("CAM-01", 2),)))
        self.assertEqual(result.results[0].detections, ())

    def test_stale_and_duplicate_protection(self):
        backend = FakeBackend(); detector = PersonDetector(CONFIG, backend)
        stale = packet("CAM-01", 1, time.time() - 1)
        self.assertEqual(detector.process_batch(BatchOutput(1, time.time(), (stale,))).results, ())
        fresh = packet("CAM-01", 2); detector.process_batch(BatchOutput(2, time.time(), (fresh,)))
        self.assertEqual(detector.process_batch(BatchOutput(3, time.time(), (fresh,))).results, ())
        metrics = detector.metrics.snapshot()
        self.assertEqual(metrics["stale_drops_before_inference"], 1)
        self.assertEqual(metrics["duplicate_inference_prevented"], 1)
        self.assertEqual(backend.calls, 1)

    def test_starved_camera_does_not_block_detection(self):
        detected, backend = [], FakeBackend()
        detector = PersonDetector(CONFIG, backend)
        scheduler = BatchScheduler(100, lambda batch: detected.append(detector.process_batch(batch)), batch_collect_window_ms=5)
        buffers = {cid: LatestFrameBuffer(scheduler.notify_frame_available) for cid in ("CAM-01", "CAM-02")}
        for cid, buffer in buffers.items(): scheduler.register_camera(cid, buffer)
        scheduler.start(); buffers["CAM-01"].put(packet("CAM-01", 7)); time.sleep(.05)
        scheduler.stop(); scheduler.join(1)
        self.assertEqual(detected[0].results[0].camera_id, "CAM-01"); self.assertEqual(backend.calls, 1)

if __name__ == "__main__": unittest.main()
