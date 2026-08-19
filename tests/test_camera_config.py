from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.camera_v2.config import load_settings


class CameraConfigTests(unittest.TestCase):
    def _write(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        try:
            tmp.write(text)
            return Path(tmp.name)
        finally:
            tmp.close()

    def test_disabled_camera_is_filtered_and_env_credentials_win(self) -> None:
        path = self._write(
            """
cameras:
  - id: CAM-01
    name: Entrance
    room: Lobby
    enabled: true
    uri: rtsp://example/101
    username: yaml-user
    password: yaml-pass
  - id: CAM-02
    enabled: false
    uri: rtsp://example/201

deepstream:
  rtsp_transport: tcp

display:
  width: 736
  height: 416
  fps: 20
  jpeg_quality: 70
"""
        )
        try:
            with patch.dict(
                os.environ,
                {
                    "SURVEILLANCE_RTSP_USERNAME": "global-user",
                    "SURVEILLANCE_RTSP_PASSWORD": "global-pass",
                    "CAM_01_RTSP_USERNAME": "camera-user",
                    "CAM_01_RTSP_PASSWORD": "camera-pass",
                },
                clear=False,
            ):
                settings = load_settings(path)
            self.assertEqual(len(settings.cameras), 1)
            camera = settings.cameras[0]
            self.assertEqual(camera.camera_id, "CAM-01")
            self.assertEqual(camera.username, "camera-user")
            self.assertEqual(camera.password, "camera-pass")
            self.assertEqual(settings.deepstream.rtsp_transport, "tcp")
        finally:
            path.unlink(missing_ok=True)

    def test_duplicate_camera_id_is_rejected(self) -> None:
        path = self._write(
            """
cameras:
  - id: CAM-01
    uri: rtsp://example/101
  - id: cam-01
    uri: rtsp://example/201
"""
        )
        try:
            with self.assertRaisesRegex(ValueError, "Duplicate camera id"):
                load_settings(path)
        finally:
            path.unlink(missing_ok=True)

    def test_non_rtsp_uri_is_rejected(self) -> None:
        path = self._write(
            """
cameras:
  - id: CAM-01
    uri: http://example/camera
"""
        )
        try:
            with self.assertRaisesRegex(ValueError, "rtsp://"):
                load_settings(path)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
