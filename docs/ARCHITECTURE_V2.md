# AI Surveillance V2 — GPU-First Clean Rebuild

Branch: `rebuild/gpu-v2-clean`

## Non-negotiable goals

1. Six RTSP cameras remain visually smooth before any AI is enabled.
2. Camera display stays GPU-native: no JPEG, MJPEG, Qt frame copies, NumPy copies, mmap, or Python per-frame drawing in the display hot path.
3. One RTSP connection and one hardware decode session per camera.
4. A slow/reconnecting camera must not freeze the other five.
5. AI is a side path. It is never allowed to block camera ingest or display.
6. We add exactly one capability at a time and keep a measured rollback baseline.

## Phase 0 — Environment and camera truth

Before building the pipeline, collect facts instead of assuming them:

- installed DeepStream/GStreamer plugin versions;
- GPU name, compute capability, driver and CUDA runtime;
- each camera's real codec, width, height, PTS cadence and FPS;
- RTSP TCP vs UDP stability;
- packet/jitter/drop statistics;
- whether the display session is X11 or Wayland;
- whether EGL/OpenGL rendering is really using NVIDIA.

The application config must not pretend a camera is 25/30 FPS if the RTSP stream actually sends 20 FPS.

## Phase 1 — Golden camera-only GPU wall

Canonical pipeline:

```text
CAM-01 nvurisrcbin/NVDEC -> queue(max=1, leaky) -> nvstreammux sink_0
CAM-02 nvurisrcbin/NVDEC -> queue(max=1, leaky) -> nvstreammux sink_1
CAM-03 nvurisrcbin/NVDEC -> queue(max=1, leaky) -> nvstreammux sink_2
CAM-04 nvurisrcbin/NVDEC -> queue(max=1, leaky) -> nvstreammux sink_3
CAM-05 nvurisrcbin/NVDEC -> queue(max=1, leaky) -> nvstreammux sink_4
CAM-06 nvurisrcbin/NVDEC -> queue(max=1, leaky) -> nvstreammux sink_5
                                                   |
                                                   v
                                          nvmultistreamtiler
                                                   |
                                                   v
                                            queue(max=1)
                                                   |
                                                   v
                                           nveglglessink
```

Rules:

- `nvurisrcbin` owns RTSP negotiation, reconnect and NVDEC.
- `drop-on-latency=true`.
- `low-latency-mode=true` only when the bitstream structure is compatible.
- `nvstreammux.batch-size=6`.
- `live-source=true`.
- mux width/height match the actual input resolution so there is no accidental extra scaling.
- `batched-push-timeout` is derived from measured source FPS, not guessed.
- `sync-inputs=false` for the first low-latency baseline so one source cannot block all cameras.
- output wall defaults to `1920x720` for a 3x2 wall; do not render an unnecessary `3840x1440` wall on a 1050 Ti.
- final queue is latest-only.
- sink starts with `sync=false`, `qos=false` for the low-latency baseline.

Acceptance criteria for Phase 1:

- all six sources stay within roughly one frame of their measured RTSP cadence;
- queue depth stays 0/1;
- no `A lot of buffers are being dropped` warnings;
- no monotonic growth in latency;
- one camera disconnect does not freeze the remaining five;
- visually smooth motion is confirmed locally, not only through AnyDesk;
- camera-only baseline runs for at least 15 minutes before AI is added.

## Phase 2 — YOLO26m person detection side path

Do not put PyTorch in series with the display pipeline.

Per camera:

```text
NVDEC -> tee
  |       \
  |        -> inference gate -> resize/convert only on demand -> appsink
  |
  -> display queue -> nvstreammux -> tiler -> EGL
```

The inference gate drops frames BEFORE expensive color conversion / host copy. Only a requested detector frame is converted.

### GTX 1050 Ti policy

Smoothness is more important than maximum detector throughput.

Do not assume `batch=6` is optimal. A single large YOLO26m batch can create a long CUDA burst that visibly stalls graphics on Pascal even when average GPU utilization is low.

Benchmark in this order:

1. micro-batch=2;
2. micro-batch=3;
3. batch=6 only if camera smoothness remains identical to Phase 1.

All six cameras are still serviced fairly using round-robin scheduling. The detector consumes the latest frame only; stale work is discarded.

Initial inference profile:

- model: YOLO26m;
- class: person only;
- input: start around 448x256 or 512x288;
- detector cadence: start low and increase only while camera metrics stay healthy;
- `torch.inference_mode()`;
- no FP16 assumption on GTX 1050 Ti until measured;
- no TensorRT 10 path on SM 6.1;
- PyTorch worker isolated from GStreamer control threads.

Acceptance criteria for Phase 2:

- camera FPS/latency stays effectively equal to Phase 1;
- detector never creates queue backlog;
- no stale inference queue;
- zero inference errors;
- person count results are correct before any bbox overlay is added.

## Phase 2.1 — Visible bbox only

Only after detection is stable:

- add object metadata / OSD;
- do not use Python/Cairo drawing in the per-frame display path;
- prefer DeepStream metadata + `nvdsosd` if available and verified;
- keep OSD removable with one config flag so its cost can be measured independently.

Acceptance criteria:

- boxes visible;
- no display FPS regression;
- no sink late-drop warnings.

## Later phases — not implemented until camera+detection is green

1. local tracker;
2. cross-camera ReID;
3. face recognition;
4. camera-space heatmap;
5. API/database;
6. final UI.

Each stage is a sidecar or metadata stage and must be benchmarked against the previous golden baseline.

## Observability built in from day one

Every five seconds print:

```text
CAM-01 src_fps pts_interval queue reconnects
...
CAM-06 src_fps pts_interval queue reconnects
WALL rendered_fps warnings
DETECT batches/s inputs/s batch_ms frame_age_ms queue_depth errors
GPU util memory decoder_util
```

Also support:

```bash
NVDS_ENABLE_LATENCY_MEASUREMENT=1
NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1
```

No optimization is accepted without before/after numbers.

## Repository rule

The old `main` implementation is kept as reference only. New V2 code lives separately and does not import old experimental `core_v1` camera/detection modules. When V2 Phase 1 and Phase 2 pass their acceptance tests, they can replace the old runtime deliberately.
