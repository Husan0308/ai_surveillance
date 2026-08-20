from __future__ import annotations

"""Compatibility entrypoint for the stable YOLO detector mode.

Detection is currently isolated deliberately: no temporal tracker and no optical
flow are allowed until raw YOLO26m person boxes are proven on the live cameras.
The old stable tracker remains in the repository and will only be layered back
on after detector truth is visually confirmed.
"""

from .stable_yolo_truth_backend import install, stable_yolo_truth_worker

__all__ = ["install", "stable_yolo_truth_worker"]
