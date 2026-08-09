import threading,time,unittest
import numpy as np
from tempfile import TemporaryDirectory
from services.api_service.database import SQLiteDatabase
from services.api_service.repositories.domain import CameraRepository
from services.ml_service.cameras.config import load_camera_configs
from services.ml_service.cameras.gstreamer import nvidia_rtsp_pipeline
from services.ml_service.runtime_control import RuntimeCommandLoop
from services.ml_service.secondary import SecondaryAIScheduler,SecondaryTask,SecondaryTaskType

class P1RuntimeTests(unittest.TestCase):
    def task(self,kind=SecondaryTaskType.FACE,stamp=None):return SecondaryTask(kind,"CAM-01","T1",None,1,stamp or time.time(),(0,0,2,2),np.zeros((2,2,3),np.uint8))
    def test_pipeline_is_codec_specific_bounded_and_nvdec(self):
        value=nvidia_rtsp_pipeline({"id":"CAM-01","source":"rtsp://host/x","codec":"h265","latency_ms":20})
        self.assertIn("rtph265depay",value);self.assertIn("nvv4l2decoder",value);self.assertIn("appsink name=sink",value);self.assertIn("max-buffers=1",value);self.assertNotIn("enable-max-performance",value)
        alternative=nvidia_rtsp_pipeline({"id":"CAM-01","source":"rtsp://host/x","codec":"h264","decoder_backend":"nvcodec"})
        self.assertIn("nvh264dec",alternative);self.assertNotIn("avdec",alternative)
    def test_commands_run_without_batches(self):
        pending=[{"type":"camera.config.changed"}];seen=threading.Event()
        loop=RuntimeCommandLoop(lambda: [pending.pop()] if pending else [],lambda _item:seen.set(),.01);loop.start()
        self.assertTrue(seen.wait(.3));loop.stop();self.assertTrue(loop.join(1))
    def test_slow_face_does_not_block_reid_and_queues_are_bounded(self):
        face_started=threading.Event();reid_done=threading.Event()
        def face(_task):face_started.set();time.sleep(.5)
        scheduler=SecondaryAIScheduler({SecondaryTaskType.FACE:face,SecondaryTaskType.REID:lambda tasks:([reid_done.set() for _ in tasks] or [None]*len(tasks))},queue_size=1,max_task_age_ms=100)
        scheduler.start();self.assertTrue(scheduler.submit(self.task()));self.assertTrue(face_started.wait(.2));self.assertTrue(scheduler.submit(self.task(SecondaryTaskType.REID)));self.assertTrue(reid_done.wait(.2))
        self.assertTrue(scheduler.submit(self.task(SecondaryTaskType.FACE)));self.assertFalse(scheduler.submit(self.task(SecondaryTaskType.FACE)))
        time.sleep(.6);self.assertGreaterEqual(scheduler.snapshot()["face"]["stale"],1);scheduler.shutdown(2);self.assertFalse(scheduler.alive_threads())

    def test_api_camera_records_override_yaml_runtime_fields(self):
        records=[{"id":"CAM-01","rtsp_url":"rtsp://db-authority/live","enabled":True,"codec":"h265"}]
        loaded=load_camera_configs(api_url="http://unused",fetcher=lambda _url:records)
        self.assertEqual(loaded[0]["source"],"rtsp://db-authority/live");self.assertEqual(loaded[0]["codec"],"h265")

class CameraPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_camera_runtime_state_survives_database_restart(self):
        with TemporaryDirectory() as directory:
            path=f"{directory}/state.db";first=SQLiteDatabase(path);self.assertTrue(await first.connect())
            repo=CameraRepository(first);await repo.update("CAM-02",{"enabled":False,"name":"Persisted camera","ai_source":"rtsp://persisted/ai","display_source":"rtsp://persisted/display"});await first.close()
            second=SQLiteDatabase(path);self.assertTrue(await second.connect());record=await CameraRepository(second).get("CAM-02")
            self.assertFalse(record["enabled"]);self.assertEqual(record["name"],"Persisted camera");self.assertEqual(record["ai_source"],"rtsp://persisted/ai");self.assertEqual(record["display_source"],"rtsp://persisted/display");await second.close()

if __name__=="__main__":unittest.main()
