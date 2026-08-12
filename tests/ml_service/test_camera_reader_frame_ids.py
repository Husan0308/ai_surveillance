import unittest
import numpy as np

from services.ml_service.cameras.buffer import LatestFrameBuffer
from services.ml_service.cameras.reader import CameraReader


class CameraReaderFrameIdTests(unittest.TestCase):
    def test_replacement_reader_continues_from_preserved_frame_id(self):
        buffer=LatestFrameBuffer()
        reader=CameraReader({"id":"CAM-01","source":0,"_initial_frame_id":100},buffer,capture_factory=lambda _cfg:None)
        frame=np.zeros((16,16,3),dtype=np.uint8)
        reader._accept(frame,1.0,1.0)
        packet=buffer.take()
        self.assertIsNotNone(packet)
        self.assertEqual(packet.frame_id,101)
        self.assertEqual(reader.metrics()["recv_frame_id"],101)

    def test_default_reader_still_starts_at_one(self):
        buffer=LatestFrameBuffer()
        reader=CameraReader({"id":"CAM-02","source":0},buffer,capture_factory=lambda _cfg:None)
        frame=np.zeros((16,16,3),dtype=np.uint8)
        reader._accept(frame,1.0,1.0)
        self.assertEqual(buffer.take().frame_id,1)


if __name__=="__main__":
    unittest.main()
