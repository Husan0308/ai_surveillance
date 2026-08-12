import threading,time,unittest
import numpy as np
from tempfile import TemporaryDirectory
from services.api_service.database import SQLiteDatabase
from services.api_service.repositories.domain import CameraRepository
from services.ml_service.cameras.config import load_camera_configs
from services.ml_service.cameras.gstreamer import jitter_nanoseconds_to_ms,nvidia_rtsp_pipeline,owned_bgr_from_mapped
from services.ml_service.runtime_control import RuntimeCommandLoop
from services.ml_service.secondary import SecondaryAIScheduler,SecondaryTask,SecondaryTaskType

class P1RuntimeTests(unittest.TestCase):
    def test_bgrx_padding_removal_is_pixel_exact_and_owned(self):
        import numpy as np
        bgr=np.arange(18,dtype=np.uint8).reshape(2,3,3);bgrx=np.empty((2,3,4),dtype=np.uint8);bgrx[...,:3]=bgr;bgrx[...,3]=255
        result=owned_bgr_from_mapped(memoryview(bgrx),3,2,"BGRx")
        np.testing.assert_array_equal(result,bgr);bgrx[0,0,0]=99;self.assertNotEqual(result[0,0,0],99)
    def test_rtp_average_jitter_is_reported_in_milliseconds(self):
        self.assertAlmostEqual(jitter_nanoseconds_to_ms(4_254_967),4.254967);self.assertIsNone(jitter_nanoseconds_to_ms(None))
    def task(self,kind=SecondaryTaskType.FACE,stamp=None):return SecondaryTask(kind,"CAM-01","T1",None,1,stamp or time.time(),(0,0,2,2),np.zeros((2,2,3),np.uint8))
    def test_pipeline_is_codec_specific_bounded_and_nvdec(self):
        value=nvidia_rtsp_pipeline({"id":"CAM-01","source":"rtsp://host/x","codec":"h265","latency_ms":20})
        self.assertIn("rtph265depay",value);self.assertIn("nvv4l2decoder",value);self.assertIn("appsink name=sink",value);self.assertIn("max-buffers=1",value);self.assertNotIn("enable-max-performance",value);self.assertIn("format=BGRx",value);self.assertNotIn("! videoconvert !",value)
        alternative=nvidia_rtsp_pipeline({"id":"CAM-01","source":"rtsp://host/x","codec":"h264","decoder_backend":"nvcodec"})
        self.assertIn("nvh264dec",alternative);self.assertNotIn("avdec",alternative)
        udp=nvidia_rtsp_pipeline({"id":"CAM-01","source":"rtsp://host/x","codec":"h264","rtsp_transport":"udp","udp_buffer_size":2097152,"latency_ms":80})
        self.assertIn("protocols=udp",udp);self.assertIn("udp-buffer-size=2097152",udp);self.assertIn("latency=80",udp)
        automatic=nvidia_rtsp_pipeline({"id":"CAM-01","source":"rtsp://host/x","codec":"h264","rtsp_transport":"auto"})
        self.assertNotIn("protocols=",automatic)
        with self.assertRaisesRegex(ValueError,"unsupported RTSP transport"):
            nvidia_rtsp_pipeline({"id":"CAM-01","source":"rtsp://host/x","codec":"h264","rtsp_transport":"invalid"})
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
    async def test_verified_cam05_display_role_migrates_without_changing_ai_source(self):
        with TemporaryDirectory() as directory:
            path=f"{directory}/state.db";database=SQLiteDatabase(path);self.assertTrue(await database.connect())
            record=await CameraRepository(database).get("CAM-05")
            self.assertTrue(record["ai_source"].endswith("/Channels/502"));self.assertTrue(record["display_source"].endswith("/Channels/501"));self.assertEqual(record["display_codec"],"h265")
            await CameraRepository(database).update("CAM-05",{"display_source":"rtsp://custom/display"});await database.close()
            restarted=SQLiteDatabase(path);self.assertTrue(await restarted.connect());custom=await CameraRepository(restarted).get("CAM-05")
            self.assertEqual(custom["display_source"],"rtsp://custom/display");self.assertTrue(custom["ai_source"].endswith("/Channels/502"));await restarted.close()

    async def test_recovery_roi_survives_database_restart(self):
        with TemporaryDirectory() as directory:
            path=f"{directory}/state.db";first=SQLiteDatabase(path);self.assertTrue(await first.connect());roi={"id":"far_desks","enabled":True,"polygon":[[.1,.1],[.8,.1],[.8,.5],[.1,.5]]};await CameraRepository(first).update("CAM-05",{"recovery_rois":[roi]});await first.close();second=SQLiteDatabase(path);self.assertTrue(await second.connect());record=await CameraRepository(second).get("CAM-05");self.assertEqual(record["recovery_rois"],[roi]);await second.close()

    async def test_camera_runtime_state_survives_database_restart(self):
        with TemporaryDirectory() as directory:
            path=f"{directory}/state.db";first=SQLiteDatabase(path);self.assertTrue(await first.connect())
            repo=CameraRepository(first);await repo.update("CAM-02",{"enabled":False,"name":"Persisted camera","ai_source":"rtsp://persisted/ai","display_source":"rtsp://persisted/display"});await first.close()
            second=SQLiteDatabase(path);self.assertTrue(await second.connect());record=await CameraRepository(second).get("CAM-02")
            self.assertFalse(record["enabled"]);self.assertEqual(record["name"],"Persisted camera");self.assertEqual(record["ai_source"],"rtsp://persisted/ai");self.assertEqual(record["display_source"],"rtsp://persisted/display");await second.close()

if __name__=="__main__":unittest.main()
