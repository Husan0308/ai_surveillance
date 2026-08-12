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

    def test_latest_frame_buffer_has_one_slot_and_replaces_obsolete(self):
        buffer=LatestFrameBuffer()
        for frame_id in range(1,7):buffer.put(self.packet("CAM-01",frame_id))
        self.assertEqual([packet.frame_id for packet in buffer.packets()],[6]);self.assertEqual(buffer.take().frame_id,6);self.assertEqual(buffer.dropped_old,5);self.assertIsNone(buffer.take())
        with self.assertRaisesRegex(ValueError,"fixed at 1"):LatestFrameBuffer(capacity=5)

    def test_slow_detector_cannot_build_batch_backlog_and_next_batch_is_newest(self):
        entered,released,second=threading.Event(),threading.Event(),threading.Event();delivered=[]
        def consume(batch):
            delivered.append(batch)
            if len(delivered)==1:entered.set();released.wait(1)
            else:second.set()
        scheduler=BatchScheduler(5000,consume,batch_collect_window_ms=0);buffer=LatestFrameBuffer(scheduler.notify_frame_available);scheduler.register_camera("CAM-01",buffer);scheduler.start()
        buffer.put(self.packet("CAM-01",1));self.assertTrue(entered.wait(1))
        for frame_id in range(2,7):buffer.put(self.packet("CAM-01",frame_id))
        self.assertEqual([packet.frame_id for packet in buffer.packets()],[6]);released.set();self.assertTrue(second.wait(1));scheduler.stop();scheduler.join(1)
        self.assertEqual([batch.frames[0].frame_id for batch in delivered],[1,6]);self.assertEqual(buffer.dropped_old,4)
        metrics=scheduler.snapshot_metrics({"CAM-01":{"online":True,"recv_frame_id":6}});self.assertEqual(metrics["scheduler_batches_created"],2);self.assertEqual(metrics["cameras"]["CAM-01"]["buffer_depth"],0)

    def test_starvation_uses_reader_decode_time_not_detector_consumption_time(self):
        scheduler=BatchScheduler(250,starved_after_ms=500);buffer=LatestFrameBuffer();scheduler.register_camera("CAM-01",buffer)
        scheduler._states["CAM-01"]["last_seen"]=time.monotonic()-10
        fresh=scheduler.snapshot_metrics({"CAM-01":{"online":True,"last_decode_timestamp":time.time(),"recv_frame_id":50}})
        self.assertFalse(fresh["cameras"]["CAM-01"]["is_starved"])
        stale=scheduler.snapshot_metrics({"CAM-01":{"online":True,"last_decode_timestamp":time.time()-1,"recv_frame_id":50}})
        self.assertTrue(stale["cameras"]["CAM-01"]["is_starved"])

    def test_partial_batch_does_not_wait_for_starved_camera(self):
        delivered, ready = [], threading.Event()
        scheduler = BatchScheduler(250, lambda batch: (delivered.append(batch), ready.set()), batch_collect_window_ms=10)
        buffers = {camera: LatestFrameBuffer(scheduler.notify_frame_available) for camera in ("CAM-01", "CAM-02", "CAM-03")}
        for camera, buffer in buffers.items(): scheduler.register_camera(camera, buffer)
        scheduler.start()
        buffers["CAM-01"].put(self.packet("CAM-01", 1)); buffers["CAM-03"].put(self.packet("CAM-03", 1))
        self.assertTrue(ready.wait(1)); scheduler.stop(); scheduler.join(1)
        self.assertEqual(delivered[0].camera_ids, ("CAM-01", "CAM-03"))

    def test_bounded_batches_are_round_robin_and_latest_only(self):
        delivered=[];ready=threading.Event()
        def consume(batch):
            delivered.append(batch)
            if len(delivered)>=2:ready.set()
        scheduler=BatchScheduler(5000,consume,batch_collect_window_ms=0,max_batch_size=3)
        buffers={camera:LatestFrameBuffer(scheduler.notify_frame_available) for camera in ("CAM-01","CAM-02","CAM-03","CAM-04","CAM-05","CAM-06")}
        for camera,buffer in buffers.items():scheduler.register_camera(camera,buffer)
        for camera,buffer in buffers.items():buffer.put(self.packet(camera,1))
        scheduler.start();self.assertTrue(ready.wait(2));scheduler.stop();scheduler.join(1)
        self.assertEqual(delivered[0].camera_ids,("CAM-01","CAM-02","CAM-03"))
        self.assertEqual(delivered[1].camera_ids,("CAM-04","CAM-05","CAM-06"))
        self.assertTrue(all(len(batch.frames)<=3 for batch in delivered))
        used=[packet.camera_id for batch in delivered[:2] for packet in batch.frames]
        self.assertEqual(len(used),len(set(used)))
        self.assertTrue(all(len(buffer)==0 for buffer in buffers.values()))

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

    def test_risk_aware_scheduler_prioritizes_danger_without_reducing_batch_capacity(self):
        scheduler=BatchScheduler(5000,batch_collect_window_ms=0,max_batch_size=3,mode="risk_aware",min_batch_size=2,fairness_deadline_ms=900)
        buffers={camera:LatestFrameBuffer() for camera in ("CAM-01","CAM-02","CAM-03","CAM-04","CAM-05","CAM-06")}
        for camera,buffer in buffers.items():scheduler.register_camera(camera,buffer);buffer.put(self.packet(camera,1))
        scheduler.update_camera_risks({"CAM-04":{"observation_age_ms":850,"lost_tracks":1,"association_ambiguity":.1}})
        batches=[]
        for frame_id in range(2,6):
            batch=scheduler._collect_locked();self.assertIsNotNone(batch);batches.append(batch)
            for camera_id in batch.camera_ids:buffers[camera_id].put(self.packet(camera_id,frame_id))
        self.assertIn("CAM-04",batches[0].camera_ids)
        self.assertTrue(all(len(batch.frames)==3 for batch in batches), "Recovery priority must not shrink or exceed configured batch=3")
        self.assertTrue(all(len(batch.camera_ids)==len(set(batch.camera_ids)) for batch in batches), "No duplicate cameras in batch")
        selected={camera_id for batch in batches[:4] for camera_id in batch.camera_ids}
        self.assertEqual(selected,set(buffers), "Zero starvation: all cameras served within fairness window")
        metrics=scheduler.snapshot_metrics({camera:{"online":True,"last_decode_timestamp":time.time()} for camera in buffers})
        self.assertEqual(metrics["scheduler_mode"],"risk_aware");self.assertEqual(metrics["scheduler_target_batch_size"],3)

    def test_adaptive_mode_expands_to_full_batch_when_recovery_is_urgent(self):
        scheduler=BatchScheduler(5000,batch_collect_window_ms=0,max_batch_size=3,mode="adaptive",min_batch_size=2,fairness_deadline_ms=900)
        buffers={camera:LatestFrameBuffer() for camera in ("CAM-01","CAM-02","CAM-03")}
        for camera,buffer in buffers.items():scheduler.register_camera(camera,buffer);buffer.put(self.packet(camera,1))
        scheduler.update_camera_risks({"CAM-02":{"observation_age_ms":900,"lost_tracks":1}})
        batch=scheduler._collect_locked();self.assertIsNotNone(batch)
        self.assertEqual(len(batch.frames),3);self.assertIn("CAM-02",batch.camera_ids)

if __name__ == "__main__": unittest.main()
