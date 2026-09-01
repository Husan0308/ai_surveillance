# Sentinel VMS — staged realtime integration

This package preserves the supplied Sentinel PySide6 interface: Monitoring, People,
Events, Rooms, Settings, the two-column monitoring grid, identity rail, fullscreen
camera grid, enrollment flow, dialogs, styles, and controls.

Stage 1 changes only the first Monitoring camera card. `CAM-01` is rendered from the
already-running V11 DeepStream/TRT8.6 pipeline through a latest-only shared-memory
preview. The preview is tapped after `nvdsosd`, so the UI receives the same live
frame and detector bbox that the stable pipeline renders. It opens no RTSP session
and runs no detector/tracker/ReID/face model.

The remaining camera cards and People/Events/Rooms data intentionally stay on the
supplied deterministic demo state until their own staged integration step. This is
deliberate: demos are removed one subsystem at a time instead of replacing the
whole UI at once.

Run the V11 pipeline first:

```bash
bash scripts/run_camera_v11_ui_cam01_pipeline_v1.sh
```

Then, in another terminal:

```bash
bash scripts/run_sentinel_ui_cam01_v1.sh
```
