import unittest
from PySide6.QtCore import QCoreApplication
from services.frontend.websocket_client import WebSocketClient
class WebSocketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QCoreApplication.instance() or QCoreApplication([])
    def test_structured_dispatch(self):
        client=WebSocketClient();received=[];client.message.connect(received.append)
        client._message('{"type":"camera.online","camera_id":"CAM-01"}')
        self.assertEqual(received[0]["camera_id"],"CAM-01");client.close()
