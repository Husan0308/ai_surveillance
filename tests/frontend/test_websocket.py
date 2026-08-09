import unittest
from PySide6.QtCore import QCoreApplication
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

class WebSocketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QCoreApplication.instance() or QCoreApplication([])
    def test_structured_dispatch(self):
        client=WebSocketClient();received=[];client.message.connect(received.append)
        client._message('{"type":"camera.online","camera_id":"CAM-01"}')
        self.assertEqual(received[0]["camera_id"],"CAM-01");client.close()
