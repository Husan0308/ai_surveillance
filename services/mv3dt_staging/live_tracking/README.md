# Entrance Room live tracking bridge (experimental)

This staging-only runner connects the existing authorized CAM-02 and CAM-05
settings to a local X11 review window. It does not enable MV3DT, MQTT, global
ID association, persistent identity, map projection, or production writes.

## Data path

`protected host camera settings + SourceOwnership lock` → host
`DeepStreamCapture` RTSP/NVDEC decode (H.264 CAM-02, H.265 CAM-05) → the
previously validated empirical `rectification_bridge.py` → bounded latest-frame
queues → packed BGR appsrc → disposable DeepStream container → BGR/NV12
conversion and NVMM → legacy `nvstreammux` (`USE_NEW_NVSTREAMMUX=no`, batch 2,
1920×1080) → selected staging detector → stock local NvDCF → exact
source/frame metadata sidecar → PySide6 on `DISPLAY=:0`.

PeopleNet Transformer remains the default rollback path. The opt-in YOLO26m
path reuses the frozen raw one-to-many FP16 model and parser with confidence
0.25, DeepStream NMS IoU 0.45, and 640×640 input. Only the PGIE batch setting is
rendered from 6 to 2; the ONNX source is omitted from the engine-only runtime
config so a plan failure cannot trigger an unapproved engine rebuild. It uses
the previously validated pinned DeepStream image digest, camera-local class ID
0 for person, and the existing NvDCF configuration. Static hashes are checked
before any camera connection; this is not detector-accuracy validation.

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
first samples device-total GPU usage three times and opens no camera if any
sample is unavailable or at/above the existing 95% overload guard. It then
takes cooperative source-owner locks for CAM-02 and CAM-05, starts both
read-only RTSP subscriptions, requires both feeds online within 30 seconds,
then performs a 15-second inference smoke. It proceeds to visual review only
when three recent device-total GPU samples are available and fewer than two
are at or above 95%; missing telemetry or sustained load stops after the
smoke. A watchdog stops frame acquisition 57 seconds after the first RTSP
connection attempt, leaving a 3-second teardown reserve inside a 60-second
authorization cap. The review is shortened to fit the remaining cap. The
private report records each camera's connection-attempt, first/last decoded
frame, capture-close-return, and camera-owner-lock-release timestamps; it
marks any actual close that exceeds the 60-second cap as a failure. Both RTSP
pipelines and the disposable DeepStream container are closed on success,
timeout, UI close, or error. This is not a service or a boot-time process.

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

## Explicit YOLO26m smoke-only command

This command uses the validated batch-2 YOLO26m candidate, displays the live
PySide6 preview, and stops after the 15-second smoke even if resources would
permit the longer review. It does not enable MV3DT or change production:

```bash
cd /home/apsidal/ai_surveillance-gate3j
DISPLAY=:0 .runtime/full-stack-venv/bin/python -B -m \
  services.mv3dt_staging.live_tracking.live_bridge live \
  --bridge /tmp/entrance-live-rectification-bridge-20261019/rectification_bridge.py \
  --layout /tmp/amc-nvidia-skill-stage.pd7zJs/projects/project_20261018_080650_1744/manual_adjustment/layout.png \
  --detector yolo26m \
  --yolo-artifacts .runtime/sixcam-approved-bbox-20261006-r12-profile/models \
  --smoke-only
```

The owner-only report contains device-total GPU/VRAM samples, per-stage
p50/p95/p99 timings, per-camera decode/processed FPS, host queue drops, exact
frame crosswalk counts, local NvDCF ID frame spans, and same-frame IoU≥0.90
box-overlap candidates (not adjudicated duplicate detections). The paint timing
is measured from appsink receipt to Qt paint, not sensor exposure; original
source PTS/DTS are retained but do not establish cross-camera synchronization.
RTSP timestamps use host UTC plus monotonic time and do not claim sensor-clock
synchronization.
Omit `--detector` to use the unchanged PeopleNet rollback path.

## Bounded native-frame visual evidence

For an explicitly authorized 30–45 second CAM-02/CAM-05 review, native-frame
mode can save a bounded, private representative sequence without rectification,
map projection, or cross-camera IDs. Each selected TRACK event is accepted only
when its source ID, input-frame index, source PTS/DTS, and appsink arrival time
match the exact raw frame in the bridge cache. The source is saved as lossless native BGR bytes;
annotated JPEGs show local NvDCF boxes/IDs, and each per-camera `frames.jsonl`
crosswalk includes both detector-space and native-space boxes plus hashes.
Directories are mode 0700 and files mode 0600. The review command below samples
every 15th decoded frame (max 80 per camera), plus a bounded sample when a
camera's local tracker-ID set changes; raw DeepStream events retain the full
event stream. Evidence enqueue is non-blocking and holds an immutable cached
frame by reference; annotation and disk writes run on a dedicated thread. Queue
overflow is explicitly counted as a dropped evidence sample and does not block
or stop inference/tracking. A metadata crosswalk mismatch is still a hard
visual-evidence failure; asynchronous disk errors or any drops make the
visual-evidence result PARTIAL, never silently accepted.

The writer passed an offline 45-second stress at 20 FPS per camera using
synthetic native-resolution frames and sparse sampling. This verifies only the
writer path—not camera access, detection, tracking, or visual accuracy. Private
metrics include queue depth, disk throughput, memory and sample integrity.

Example only—requires a fresh, explicit authorization before use:

```bash
cd /home/apsidal/ai_surveillance-gate3j
DISPLAY=:0 .runtime/full-stack-venv/bin/python -B -m \
  services.mv3dt_staging.live_tracking.live_bridge live \
  --native-frame --save-visual-evidence \
  --detector yolo26m \
  --yolo-artifacts .runtime/sixcam-approved-bbox-20261006-r12-profile/models \
  --smoke-seconds 15 --review-seconds 20 \
  --capture-stop-seconds 42 --authorized-max-seconds 45 \
  --evidence-sample-every 15 --evidence-max-samples-per-camera 80 \
  --output .runtime/entrance-live-visual-validation-NEW-RUN-ID
```

Evidence capture was validated offline with lossless raw-pixel/hash checks,
metadata crosswalk rejection tests, and a bounded two-camera writer-rate test.
No RTSP retry is automatic; use a unique empty output directory for each
authorized run.
