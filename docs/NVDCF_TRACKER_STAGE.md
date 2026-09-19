# NvDCF per-camera tracker stage

Branch: `rebuild/deepstream91-nvdcf-v1-20260919`

Starting point: frozen YOLO26m detector stage after the 660-second soak and
six-camera detector isolation/recovery gates passed.

## Scope

This stage adds only per-camera multi-object tracking:

```text
6 RTSP
 -> NVDEC/NVMM
 -> nvstreammux(batch=6)
 -> nvinfer(YOLO26m raw OTM, FP16, interval=0)
 -> DeepStream NMS(conf=0.25, IoU=0.45)
 -> nvtracker(NvDCF_perf, 960x544, GPU, batch processing)
 -> tiler
 -> nvdsosd(person ID=<object_id>)
 -> NVENC/MKV + live ffplay
```

Not enabled in this stage:

- cross-camera ReID;
- face recognition;
- global identity merge;
- pose;
- heatmap.

The low-level tracker is
`/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so`
with the pinned DeepStream image's
`config_tracker_NvDCF_perf.yml`. The perf profile is deliberately used instead
of the accuracy profile so this milestone does not pull the Re-ID model into the
tracker stage. Tracker input resolution is 960x544; both values are multiples of
32.

## Evidence

The tracker probe runs after `nvtracker`. It writes
`.runtime/<run>/tracks.jsonl` with source ID, frame, PTS, tracker object ID,
detector confidence, NvDCF tracker confidence, and tracked bounding box.

`CAM-XX TRACK` telemetry reports:

- tracked frames;
- output person objects;
- untracked objects;
- duplicate IDs within one frame;
- unique IDs seen on that source.

The checker also preserves the detector gates: source FPS/queue/error counters,
YOLO parser health, NMS overlap evidence, recording validation, RSS, and
process-specific GPU memory.

## 60-second gate

Run:

```bash
python3 -m unittest tests.test_yolo26m_tracker_gate -v

python3 scripts/validate_yolo26m_tracker.py \
  --duration 60 \
  --out .runtime/yolo26m-nvdcf-60s-v1

python3 scripts/yolo26m_tracker/check_gate.py \
  .runtime/yolo26m-nvdcf-60s-v1
```

The first checker run is expected to remain BLOCKED only for the separate visual
review if automated evidence is clean.

Visually verify:

- all six tiles advance;
- one continuously visible person keeps the same ID;
- nearby people retain separate IDs;
- no obvious rapid ID flicker/switching;
- tracked boxes stay aligned.

Then:

```bash
python3 scripts/yolo26m_tracker/mark_visual_review.py \
  .runtime/yolo26m-nvdcf-60s-v1 \
  --pass-review \
  --notes "6 cameras checked; continuous people keep stable IDs; nearby people remain separate."

python3 scripts/yolo26m_tracker/check_gate.py \
  .runtime/yolo26m-nvdcf-60s-v1
```

Do not begin ReID or any later stage until this tracker gate passes.
