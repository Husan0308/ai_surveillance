"""Detection + camera-local tracking Core v1 runtime.

Only camera ingest, YOLO person detection, lightweight per-camera ByteTrack-style
tracking, JPEG publication and runtime telemetry belong to this stage. Pose,
Heatmap, ReID, face recognition and cross-camera identity remain intentionally
absent until this six-camera baseline is proven stable on the target machine.
"""

__all__ = []
