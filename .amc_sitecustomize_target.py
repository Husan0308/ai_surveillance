"""Temporary path-scoped AMC media-I/O compatibility for two recorded-video projects.

This does not modify the PyArmor-protected AMC rectifier/model runner. It routes
only these projects' H.264 reads through the image's NVIDIA decoder and only
their rectified.mp4 writes through its NVIDIA encoder.
"""
import os
import sys
from pathlib import Path

import cv2

PROJECT_IDS = (
    "project_20261002_094836_4326",  # retain the existing diagnostic scope
    "project_20261016_055304_8625",  # the authorized repair target
)
PROJECT_ROOTS = tuple(
    (pid, Path("/home/auto-calibration-ms/server/projects") / pid)
    for pid in PROJECT_IDS
)

_capture = cv2.VideoCapture
_writer = cv2.VideoWriter


def _project_root(value):
    if not isinstance(value, (str, Path)):
        return None
    path = Path(value).resolve()
    for _pid, root in PROJECT_ROOTS:
        if path.is_relative_to(root.resolve()):
            return root
    return None


def _decode(path):
    quoted = str(path).replace("\\", "\\\\").replace('"', '\\"')
    return (
        f'filesrc location="{quoted}" ! qtdemux ! h264parse ! '
        "nvv4l2decoder ! nvvideoconvert ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink sync=false"
    )


def capture(*args, **kwargs):
    if args and _project_root(args[0]) and str(args[0]).endswith(".mp4") and Path(args[0]).is_file():
        result = _capture(_decode(Path(args[0]).resolve()), cv2.CAP_GSTREAMER)
        if not result.isOpened():
            raise RuntimeError(f"AMC NVIDIA decode failed: {args[0]}")
        return result
    return _capture(*args, **kwargs)


def writer(*args, **kwargs):
    if args and str(args[0]).endswith("/rectified.mp4"):
        root = _project_root(args[0])
        if root is not None:
            filename = str(args[0])
            camera_dir = Path(filename).parent.name
            original = root / f"{camera_dir}.mp4"
            source = capture(str(original))
            fps = source.get(cv2.CAP_PROP_FPS)
            source.release()
            if fps != 20.0:
                raise RuntimeError(f"AMC expected 20 FPS source, got {fps}")
            size = args[3]
            quoted = filename.replace("\\", "\\\\").replace('"', '\\"')
            pipeline = (
                "appsrc ! videoconvert ! video/x-raw,format=BGRx ! "
                "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! "
                "nvv4l2h264enc control-rate=2 constqp=0:0:0 ! h264parse ! "
                f'qtmux ! filesink location="{quoted}"'
            )
            output = _writer(pipeline, cv2.CAP_GSTREAMER, 0, fps, size, True)
            print(
                f"AMC_NVENC_COMPAT project={root.name} source_fps={fps} "
                f"requested_fps={args[2]} size={size} opened={output.isOpened()}",
                flush=True,
            )
            if not output.isOpened():
                raise RuntimeError("AMC NVIDIA H.264 encoder failed to open")

            class PackedWriter:
                def __init__(self):
                    self.count = 0

                def write(self, frame):
                    import numpy as np

                    image = np.asarray(frame)
                    if image.dtype != np.uint8 or image.shape != (size[1], size[0], 3):
                        raise RuntimeError(
                            f"Unexpected rectified frame: {image.shape} {image.dtype}"
                        )
                    packed = np.ascontiguousarray(image)
                    self.count += 1
                    return output.write(packed)

                def __getattr__(self, name):
                    return getattr(output, name)

            return PackedWriter()
    return _writer(*args, **kwargs)


cv2.VideoCapture = capture
cv2.VideoWriter = writer

if any(pid in str(argument) for pid in PROJECT_IDS for argument in sys.argv):
    sys.path.append("/home/auto-calibration-ms/core/rectification")
