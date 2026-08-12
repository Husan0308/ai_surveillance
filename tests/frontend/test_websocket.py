import unittest,time
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QImage
from services.frontend.websocket_client import WebSocketClient
from services.frontend.video_transport import MJPEGClient
class FakeReply:
    def __init__(self,finished=None):self.finished=finished;self.aborts=0;self.deletes=0
    def abort(self):
        self.aborts+=1
        if self.finished:self.finished(self)
    def deleteLater(self):self.deletes+=1

class VideoShutdownTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):cls.app=QCoreApplication.instance() or QCoreApplication([])
    def client(self):return MJPEGClient("CAM-01","http://127.0.0.1:9/video")
    def test_stop_without_reply_and_repeated_stop(self):
        client=self.client();client.stop();client.stop();self.assertFalse(client._running);self.assertFalse(client.retry.isActive())
    def test_abort_callback_cannot_clear_local_cleanup(self):
        client=self.client();reply=FakeReply(client._finished);client.reply=reply;client._running=True;client.stop();client.stop();self.assertEqual(reply.aborts,1);self.assertEqual(reply.deletes,1);self.assertIsNone(client.reply)
    def test_stop_cancels_pending_retry(self):
        client=self.client();client._running=True;client.retry.start();client.stop();self.assertFalse(client.retry.isActive());client._reconnect();self.assertIsNone(client.reply)

    def test_paced_presenter_emits_only_a_new_latest_frame(self):
        client=self.client();received=[];client.frame.connect(lambda _camera,frame_id,_timestamp,_image:received.append(frame_id))
        image=QImage(2,2,QImage.Format_RGB32)
        client._prepared.put(("CAM-01",1,time.time(),image,time.monotonic()));client._prepared.put(("CAM-01",2,time.time(),image,time.monotonic()))
        client.last_frame=time.monotonic();client._emit_latest()
        self.assertEqual(received,[2]);self.assertEqual(client.prepared_frame_slot_max,0)
        client._emit_latest();self.assertEqual(received,[2]);self.assertEqual(client.duplicate_rendered_frame_total,0)
        metrics=client.runtime_metrics()
        self.assertEqual(metrics["newer_frame_waiting_while_duplicate_rendered_total"],0);self.assertLessEqual(client._raw.depth(),1);self.assertLessEqual(client._prepared.depth(),1)
        self.assertEqual(client.render_timer.interval(),round(1000/client.presentation_fps))

    def test_six_camera_render_timers_are_phase_staggered(self):
        clients=[MJPEGClient(f"CAM-{index:02d}","http://127.0.0.1:9/video") for index in range(1,7)]
        phases=[client.render_phase_ms for client in clients]
        interval=clients[0].render_interval_ms
        self.assertEqual(len(set(phases)),6);self.assertEqual(phases,[index*interval//6 for index in range(6)])
        self.assertTrue(all(0<=phase<clients[0].render_interval_ms for phase in phases))

class WebSocketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QCoreApplication.instance() or QCoreApplication([])
    def test_structured_dispatch(self):
        client=WebSocketClient();received=[];client.message.connect(received.append)
        client._message('{"type":"camera.online","camera_id":"CAM-01"}')
        self.assertEqual(received[0]["camera_id"],"CAM-01");client.close()
