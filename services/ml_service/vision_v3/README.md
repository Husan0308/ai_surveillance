# Vision V3 ML Camera Core

This package is the new single production camera owner.

Current phase contains only the six-camera GPU-native DeepStream baseline:

```text
6 RTSP -> nvurisrcbin/NVDEC -> latest-only source queues
       -> nvstreammux(batch=6, sync-inputs=0)
       -> 3x2 nvmultistreamtiler
       -> latest-only wall queue
       -> nveglglessink(sync=0, qos=0)
```

Run from repository root:

```bash
bash scripts/run_vision_v3_camera_core.sh
```

The launcher first runs `scripts/preflight_vision_v3_camera_core.py`.

Do not add RF-DETR-S, NvDCF, ReID, face recognition or UI integration here until the six-camera core passes the real-machine 30–60 minute smoothness/latency soak gate. RF-DETR-S is Phase 2 and will be added as a non-blocking detector branch.
