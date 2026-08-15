from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from shared.mmap_frame import MmapFrameReader, MmapFrameWriter, frame_path


class MmapFrameTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_dir = os.environ.get("AI_SURVEILLANCE_FRAME_DIR")
        os.environ["AI_SURVEILLANCE_FRAME_DIR"] = self.temp.name

    def tearDown(self):
        if self.old_dir is None:
            os.environ.pop("AI_SURVEILLANCE_FRAME_DIR", None)
        else:
            os.environ["AI_SURVEILLANCE_FRAME_DIR"] = self.old_dir
        self.temp.cleanup()

    def test_double_buffer_publishes_only_complete_latest_frame(self):
        writer = MmapFrameWriter("CAM-01", 8, 4, channels=3)
        reader = MmapFrameReader("CAM-01")
        try:
            self.assertTrue(reader.attach())
            first = np.zeros((4, 8, 3), dtype=np.uint8)
            first[:, :, 2] = 33
            meta1 = writer.write(first, 10, 1.25)
            packet1 = reader.snapshot()
            self.assertIsNotNone(packet1)
            self.assertEqual(packet1.sequence, meta1["sequence"])
            self.assertEqual((packet1.width, packet1.height, packet1.channels), (8, 4, 3))
            self.assertEqual(packet1.frame_id, 10)
            self.assertEqual(packet1.payload, first.tobytes())
            self.assertIsNone(reader.snapshot(packet1.sequence))

            second = np.full((4, 8, 3), 177, dtype=np.uint8)
            meta2 = writer.write(second, 11, 1.30)
            packet2 = reader.snapshot(packet1.sequence)
            self.assertIsNotNone(packet2)
            self.assertGreater(meta2["sequence"], meta1["sequence"])
            self.assertEqual(packet2.frame_id, 11)
            self.assertEqual(packet2.payload, second.tobytes())
            self.assertTrue(reader.mapping_is_current())
        finally:
            reader.close()
            writer.close(unlink=True)
        self.assertFalse(Path(frame_path("CAM-01")).exists())


if __name__ == "__main__":
    unittest.main()
