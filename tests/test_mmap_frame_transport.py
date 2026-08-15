from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from shared.mmap_frame import MmapFrameReader, frame_path
from shared.safe_mmap_frame import SigbusSafeMmapFrameWriter


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
        writer = SigbusSafeMmapFrameWriter("CAM-01", 8, 4, channels=3)
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

    def test_backend_restart_atomically_replaces_inode_without_truncating_old_mapping(self):
        first_writer = SigbusSafeMmapFrameWriter("CAM-02", 8, 4, channels=3)
        reader = MmapFrameReader("CAM-02")
        second_writer = None
        try:
            first_image = np.full((4, 8, 3), 41, dtype=np.uint8)
            first_writer.write(first_image, 1, 1.0)
            self.assertTrue(reader.attach())
            packet = reader.snapshot()
            self.assertIsNotNone(packet)
            old_inode = os.fstat(first_writer._fd).st_ino

            # Creating a replacement writer must not O_TRUNC the inode still
            # mapped by reader. Existing mapping remains readable until the UI
            # notices the path inode changed and re-attaches.
            second_writer = SigbusSafeMmapFrameWriter("CAM-02", 8, 4, channels=3)
            new_inode = os.fstat(second_writer._fd).st_ino
            self.assertNotEqual(old_inode, new_inode)
            self.assertFalse(reader.mapping_is_current())
            old_packet = reader.snapshot()
            self.assertIsNotNone(old_packet)
            self.assertEqual(old_packet.payload, first_image.tobytes())

            reader.close()
            self.assertTrue(reader.attach())
            second_image = np.full((4, 8, 3), 219, dtype=np.uint8)
            second_writer.write(second_image, 2, 2.0)
            new_packet = reader.snapshot()
            self.assertIsNotNone(new_packet)
            self.assertEqual(new_packet.payload, second_image.tobytes())
        finally:
            reader.close()
            if second_writer is not None:
                second_writer.close(unlink=True)
            first_writer.close(unlink=True)

    def test_preallocation_failure_is_python_error_before_mmap_access(self):
        if not hasattr(os, "posix_fallocate"):
            self.skipTest("posix_fallocate unavailable")
        with mock.patch("shared.safe_mmap_frame.os.posix_fallocate", side_effect=OSError(28, "no space")):
            with self.assertRaises(OSError):
                SigbusSafeMmapFrameWriter("CAM-03", 8, 4, channels=3)
        self.assertFalse(Path(frame_path("CAM-03")).exists())

    def test_runtime_writer_never_truncates_public_target(self):
        source = (Path(__file__).resolve().parents[1] / "shared/safe_mmap_frame.py").read_text(encoding="utf-8")
        self.assertIn("posix_fallocate", source)
        self.assertIn("os.replace", source)
        self.assertNotIn("os.O_TRUNC", source)


if __name__ == "__main__":
    unittest.main()
