# Entrance Room live tracking bridge (experimental)

This staging-only runner connects the existing authorized CAM-02 and CAM-05
settings to a local X11 review window. It does not enable MV3DT, MQTT, global
ID association, persistent identity, map projection, or production writes.

## Data path

`protected host camera settings + SourceOwnership lock` → host
`DeepStreamCapture` RTSP/NVDEC decode (H.264 CAM-02, H.265 CAM-05) → the
previously validated empirical `rectification_bridge.py` → bounded latest-frame
queues → packed BGR appsrc → disposable DeepStream 9.1 container → BGR/NV12
conversion and NVMM → legacy `nvstreammux` (`USE_NEW_NVSTREAMMUX=no`, batch 2,
1920×1080) → existing PeopleNet Transformer → stock NvDCF → exact source/frame
metadata sidecar → PySide6 on `DISPLAY=:0`.

Host source PTS/DTS and monotonic arrival time are retained as sidecar fields.
appsrc uses increasing pipeline-running-time timestamps; independent camera
clocks are not synchronized or represented as synchronized exposure time.
The displayed map is intentionally non-interactive and has no projected
markers: calibration geometry and cross-camera timing are not validated.
NvDCF IDs are camera-local. Re-ID, MV3DT global IDs and persistent-person IDs
are disabled.

## Recorded-frame appsrc regression

Before a live run, the isolated appsrc/mux contract can be exercised with the
existing staged rectified clips (no camera connection):

```bash
cd /home/apsidal/ai_surveillance-gate3j
DISPLAY=:0 .runtime/full-stack-venv/bin/python -m \
  services.mv3dt_staging.live_tracking.live_bridge mux-smoke \
  --cam02-video /tmp/amc-nvidia-skill-stage.pd7zJs/projects/project_20261018_080650_1744/cam_00.mp4 \
  --cam05-video /tmp/amc-nvidia-skill-stage.pd7zJs/projects/project_20261018_080650_1744/cam_01.mp4 \
  --frames 8 --inference
```

This test already passed in the isolated DeepStream 9.1 image. It is a recorded
clip test, not live evidence.

## One bounded live session

Only run when the room/camera recording authorization is current. The program
first takes cooperative source-owner locks for CAM-02 and CAM-05, starts both
read-only RTSP subscriptions, requires both feeds online within 30 seconds,
then performs a 15-second inference smoke. It proceeds to a 60-second visual
review only when three recent device-total GPU samples are available and
fewer than two are at or above 95%; missing telemetry or sustained load stops
after the smoke. It closes both RTSP pipelines and the disposable DeepStream
container on success, timeout, UI close, or error. The durations are fixed in
the CLI; this is not a service or a boot-time process.

```bash
cd /home/apsidal/ai_surveillance-gate3j
DISPLAY=:0 .runtime/full-stack-venv/bin/python -m \
  services.mv3dt_staging.live_tracking.live_bridge live \
  --bridge /tmp/entrance-live-rectification-bridge-20261019/rectification_bridge.py \
  --layout /tmp/amc-nvidia-skill-stage.pd7zJs/projects/project_20261018_080650_1744/manual_adjustment/layout.png
```

The command creates a new owner-only temporary output directory and prints its
path in the final JSON. The private output directory contains `live-run.json`, `live-events.jsonl`,
`deepstream.stdout.log`, and `deepstream.stderr.log`. Files are private to the
current user. `live-events.jsonl` preserves raw DeepStream metadata and never
rewrites tracker IDs. The log may include camera observations, so keep the
directory private.

The pinned disposable image is
`nvcr.io/nvidia/deepstream:9.1-triton-multiarch`, image ID
`sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b`.
It runs with `--network none`, no host ports, read-only application/model
mounts, and no MQTT or project/production data mounts. The RTSP decode remains
on the host using the existing capture service's isolated pipeline; only
rectified BGR frames and non-secret sidecar metadata enter the container.

This is an experimental integration, not an NVIDIA-supported AMC/MV3DT live
deployment path. It does not assert calibrated map geometry, cross-camera
identity, source-clock synchronization, or persistent identity.
