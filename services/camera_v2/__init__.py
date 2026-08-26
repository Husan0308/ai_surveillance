"""Clean Pascal Camera V2 runtime.

Production camera presentation is independent from detector/tracker cadence:
RTSP/NVDEC is decoded once, then a tee feeds display, NvDCF analytics and sparse
TensorRT 8.6 detector branches. Optional ReID/global identity utilities are kept
outside the camera hot path until the local camera acceptance test passes.
"""

from __future__ import annotations
